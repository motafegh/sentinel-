# 2026-05-09 — M1 ML Deep Dive

A complete M1 breakdown: process, design, architecture, decisions, flow,
data, files, assumptions, expectations, predictions, and acceptance
criteria. Companion to:

- `2026-05-09-M1-ml-plan.md`
- `2026-05-09-M1-autoresearch-integration-plan.md`
- `2026-05-09-project-state-synthesis.md` (cross-module briefing)

Verified against source state in `ml/` on 2026-05-09 and locked
constants from `docs/Project-Spec/SENTINEL-INDEX.md` /
`SENTINEL-M1-ML.md` / `SENTINEL-EVAL-BACKLOG.md`.

---

## 1. What's going to happen on M1 (in execution order)

```
1. Validate that v3 dataset still reproduces v3 numbers          (1 hr)
   └─ poetry run python ml/scripts/validate_graph_dataset.py     → exit 0
   └─ confirm val_indices.npy hash; commit val_indices.sha256 sidecar

2. Build autoresearch harness                                    (1–2 days)
   ├─ ml/scripts/auto_experiment.py        (smoke + confirm regimes)
   ├─ ml/autoresearch/program.md           (skill file)
   ├─ ml/autoresearch/README.md            (operator doc)
   └─ locked-files hash guard inside auto_experiment.py

3. Sanity reproduction                                           (1 hr)
   └─ run --regime confirm with v3 hyperparams; expect tuned F1 ≈ 0.5069 ± 0.01

4. v4 search session(s)                                          (1–3 nights)
   └─ smoke runs to find directions; confirm runs to validate winners
   └─ stop on: 12 confirms, OR 3 below-gate streak, OR human interrupt

5. Promote winner                                                (1 hr)
   ├─ tune_threshold.py final pass on full val
   ├─ promote_model.py --stage Staging --val-f1-macro <x>
   ├─ docs/STATUS.md "Active Checkpoint" updated
   └─ new dated changelog written

6. Open-audit hardening (background, after v4)                   (0.5–1 day)
   └─ Audit #9, #10/#12, #14, #15, #16, #17, #18 — see §10
```

---

## 2. Design

### 2.1 Architectural design (frozen for v4)

```
Solidity source (≤500 KB, UTF-8)
        │
        ▼
┌────────────────────────────────────┐
│  Slither AST → graph_extractor.py  │
│  → PyG Data:                       │
│     x          [N, 8]              │
│     edge_index [2, E]              │
│     edge_attr  [E] (1-D, LOCKED)   │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐  ┌──────────────────────────────┐
│  GNNEncoder                        │  │  CodeBERT + LoRA r=8 α=16    │
│  GAT, edge_emb=Embedding(5,16)     │  │  windowed if > 512 tokens    │
│  in=8 (LOCKED), use_edge_attr=True │  │  target_modules: query, value│
│  out=128                           │  │  out=768 → projected to 128  │
└──────────────┬─────────────────────┘  └──────────────┬───────────────┘
               │                                       │
               └───────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────┐
                │  CrossAttentionFusion            │
                │  bidirectional cross-attn        │
                │  output_dim = 128 (LOCKED ADR-025)│
                └──────────────┬───────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────┐
                │  Classifier head Linear(128→10)  │
                │  raw logits per class            │
                └──────────────┬───────────────────┘
                               │
                               ▼
                ┌──────────────────────────────────┐
                │  Sigmoid + per-class thresholds  │
                │  thresholds: list[10] from JSON  │
                │  (NOT a single float — Fix #6)   │
                └──────────────────────────────────┘
```

**Locked dimensions** (any change cascades into M2 CIRCUIT_VERSION
bump, ONNX re-export, full EZKL rebuild, `ZKMLVerifier.sol` redeploy
on Sepolia, and `AuditRegistry.upgradeToAndCall` to swap verifier):

```
in_channels       = 8         (GNN node features)
fusion_output_dim = 128       (CrossAttentionFusion → ProxyMLP input)
NUM_CLASSES       = 10        (append-only; WeakAccessMod excluded)
edge_attr shape   = [E] 1-D
ARCHITECTURE      = "cross_attention_lora"
```

### 2.2 Process design

Two separate flows, both rooted in M1:

**Inference flow** (per request, what M6 will eventually call):

```
source.sol
  → ContractPreprocessor.process_source()       (cached on content hash)
  → graph + tokens                              (.pt in RAM cache, TTL via mtime)
  → SentinelModel.forward()                     (windowed if needed)
  → per-class probs                             [10]
  → apply thresholds (per-class)                → bool[10]
  → emit response: {vulnerabilities, thresholds, windows_used, …}
```

**Training flow** (offline, what v4 / autoresearch use):

```
TrainConfig (dataclass) → train()
  → DualPathDataset (graph + tokens, splits)
  → DataLoader (PyG-aware collate, num_workers=2 + pin_memory)
  → forward + AMP (autocast bf16 on Ampere)
  → loss: BCE | Focal (γ, α)
  → backward + GradScaler + clip_grad_norm_(trainable only)
  → OneCycleLR
  → epoch end: validate, save best, update patience sidecar JSON
  → MLflow log: focal_γ, focal_α, val F1-macro, per-class F1
  → tune_threshold.py: per-class thresholds → JSON sidecar
```

### 2.3 Autoresearch design (the v4 search loop)

```
operator points agent at ml/autoresearch/program.md
        │
        ▼
loop (until 12 confirms / 3 below-gate streak / Ctrl-C):
   git checkout -b autoresearch/<tag>-<idx>
   (optionally edit ml/src/training/trainer.py — single allowed file)
   poetry run python ml/scripts/auto_experiment.py --regime smoke <knobs>
        │
        ▼
   pre-flight:
     validate_graph_dataset.py   → 0
     val_indices.sha256 matches  → ok
     locked-file hashes match    → ok
     GPU free ≥ 7 GB             → ok
        │
        ▼
   train (~5 min on 10% subsample)
   tune threshold (--max-batches 100)
   emit:  SENTINEL_SCORE=<float>
          PEAK_VRAM_MB=<int>
          REGIME=smoke
        │
        ▼
   if smoke > smoke_best_so_far:
       --regime confirm   (~30–60 min, full val)
       if confirm > 0.5069 AND no per-class floor breach:
           keep branch; update best pointer
       else:
           git reset --hard; delete branch
   else:
       git reset --hard; delete branch
```

---

## 3. Decisions (taken or to be taken)

### Already taken (locked)

| Decision | Source |
|---|---|
| ARCHITECTURE = `cross_attention_lora` for v4 (comparable to v3) | M1 plan §1, §2.1 |
| Use focal loss + LoRA r=16 + DoS weighted sampler as starting point | ROADMAP §"In Progress" §1, M1 plan §2.2 |
| Same `val_indices.npy` as v3 (no regeneration) | EVAL-BACKLOG §"Active Baseline" |
| Per-class threshold tuning (JSON sidecar) is mandatory | INDEX §"Critical Cross-Cutting Rules" |
| API response key is `thresholds` (list), not `threshold` | Fix #6, M1 plan §1 (breaking change) |
| No "confidence" field anywhere | Track 3 removal, INDEX §"Critical" |
| Port autoresearch *pattern*, don't fork either upstream | autoresearch plan §1.3 |
| Two-tier search (smoke / confirm) for 8 GB VRAM | autoresearch plan §4 |

### Open decisions (operator-owned)

| Decision | Default if not chosen | Owner |
|---|---|---|
| When to start v4 (now vs after Foundry bring-up) | now (M1 is on the critical path) | operator |
| First search tag (e.g. `2026-05-10`) | not started until chosen | operator |
| Whether to graduate a smoke-best to confirm if just under threshold | autoresearch plan §10 says smoke > 0.42 → confirm; calibrate after 5 confirms | operator |
| Which knob to vary first (focal γ, LoRA r, DoS sampler) | M1 plan §2.4 ranks them; agent will explore in priority order | operator / agent |
| Whether to merge winning autoresearch branch back into the working branch | manual, after promote_model | operator |

---

## 4. Flow (data + control, end-to-end for v4)

```
v3 checkpoint + thresholds
       │
       ▼ (same data, same split, same arch)
┌─────────────────────────────────────────────────────────┐
│ 1. validate_graph_dataset.py                            │
│    - reads ml/data/graphs/*.pt                          │
│    - asserts edge_attr shape [E], all 68,523 files OK   │
│    - exit 0 / 1                                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. auto_experiment.py --regime smoke                    │
│    - hashes locked files; aborts on mismatch            │
│    - builds TrainConfig(focal_γ=2.0, lora_r=16, …)      │
│    - calls train() with subsample sampler               │
│    - 1 epoch on ~4,800 graphs (10% stratified)          │
│    - validates on full val_indices                      │
│    - tune_threshold.py with --max-batches 100           │
│    - emits SENTINEL_SCORE=<smoke_tuned_F1>              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼ (only if smoke beats current best)
┌─────────────────────────────────────────────────────────┐
│ 3. auto_experiment.py --regime confirm                  │
│    - same code path, full data, ~5 epochs               │
│    - tune_threshold.py over full val                    │
│    - per-class F1 logged to MLflow                      │
│    - emits SENTINEL_SCORE=<confirmed_tuned_F1>          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼ (only if confirm > 0.5069 AND floors held)
┌─────────────────────────────────────────────────────────┐
│ 4. promote_model.py                                     │
│    - logs ckpt to MLflow as artifact                    │
│    - registers to model registry                        │
│    - transitions to Staging                             │
│    - writes audit-trail tags (val_f1, gates passed)     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
                docs/STATUS.md updated
                docs/changes/<date>-v4-training-complete.md
```

---

## 5. Data

### 5.1 Dataset facts (frozen for v3 + v4)

```
Source corpus       BCCC-SCsVul-2024
Total graphs        68,523 .pt files (md5-named)
Total tokens        68,568 .pt files
Splits (FIXED)      train: 47,966
                    val:   10,278   ← gate evaluator
                    test:  10,279
Multilabel index    ml/data/multilabel_index.csv
Class count         10  (NUM_CLASSES, locked)
```

### 5.2 Class distribution & v3 floors

| Class | Train support | v3 tuned F1 | v4 floor |
|---|---:|---:|---:|
| IntegerUO | 5,343 | 0.8214 | 0.7714 |
| GasException | 2,589 | 0.5501 | 0.5001 |
| Reentrancy | 2,501 | 0.5362 | 0.4862 |
| MishandledException | 2,207 | 0.4916 | 0.4416 |
| UnusedReturn | 1,716 | 0.4860 | 0.4360 |
| Timestamp | 1,077 | 0.4789 | 0.4289 |
| TransactionOrderDependence | ~ | 0.4770 | 0.4270 |
| ExternalBug | ~ | 0.4345 | 0.3845 |
| CallToUnknown | ~ | 0.3936 | 0.3436 |
| **DenialOfService** | **137** | 0.4000 | 0.3500 |

The 39× imbalance between IntegerUO and DenialOfService is the
single most important driver of the macro-F1 ceiling.

### 5.3 What the model actually sees per request

```
Graph node features (8-dim):
  type one-hot           (Function, Modifier, Variable, …)
  visibility one-hot     (public, private, internal, external)
  is_payable             bool
  is_view                bool
  state_mutating         bool
  is_external_call       bool
  has_assembly           bool
  has_unchecked_block    bool

Edge types (5):
  control_flow, data_flow, call, modifies_state, inherits_from
  → embedded via nn.Embedding(5, 16)

Tokens:
  CodeBERT vocab, max_seq_len=512
  windowed sliding inference for > 512 (T1-C); windows_used in response
```

### 5.4 Caches

```
ml/data/graphs/_cache/        InferenceCache (TTL via mtime)
RAM cache                     ContractPreprocessor (Audit #11 fixed: integrity-checked)
Token cache                   per content hash
```

---

## 6. Files (the complete M1 inventory)

### 6.1 New files this plan adds

```
ml/scripts/auto_experiment.py                      Single-run wrapper, smoke + confirm
ml/autoresearch/program.md                         Skill file for the agent
ml/autoresearch/README.md                          Operator doc
ml/autoresearch/results.tsv                        Runtime ledger (untracked)
ml/data/splits/val_indices.sha256                  Pin v3 split hash
ml/data/splits/locked_files.sha256                 Pin architecture/data files (hash guard)
docs/changes/<date>-v4-training-complete.md        Post-v4 changelog
```

### 6.2 Existing files M1 v4 *touches* (write)

```
ml/checkpoints/multilabel-v4-<runname>_best.pt                    new ckpt
ml/checkpoints/multilabel-v4-<runname>_best_thresholds.json       new thresholds
ml/checkpoints/multilabel-v4-<runname>_best.state.json            patience sidecar
ml/data/splits/                                                   READ-ONLY (no regen)
mlruns.db                                                         MLflow tracking
docs/STATUS.md                                                    Active Checkpoint section
```

### 6.3 Existing files M1 v4 *reads* and may modify (autoresearch agent)

```
ml/src/training/trainer.py             ← only file the agent may edit
ml/src/training/focalloss.py           ← agent reads but doesn't modify (Audit #15 doc fix is separate)
ml/scripts/train.py                    ← read; CLI args may extend
ml/scripts/tune_threshold.py           ← read; may be invoked programmatically
```

### 6.4 LOCKED — hash-pinned, agent must not modify

```
ml/src/models/sentinel_model.py        SentinelModel (CrossAttentionFusion=128)
ml/src/models/gnn_encoder.py           GAT + edge_emb (in=8)
ml/src/models/transformer_encoder.py   CodeBERT+LoRA wrapper
ml/src/models/fusion_layer.py          Legacy 64-dim fuser; not used in v4
ml/src/preprocessing/graph_schema.py   NUM_EDGE_TYPES=5, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
ml/src/preprocessing/graph_extractor.py multi_contract_policy enum
ml/data/graphs/                        68,523 .pt files
ml/data/tokens/                        68,568 .pt files
ml/data/splits/val_indices.npy         the gate
ml/data/splits/train_indices.npy       same seed, same files
```

Any hash mismatch → `auto_experiment.py` exits 1 before training.

### 6.5 Operational scripts (already exist, not changed)

```
ml/scripts/validate_graph_dataset.py    pre-flight check
ml/scripts/build_multilabel_index.py    one-time, already run
ml/scripts/create_splits.py             one-time, already run (DO NOT RUN)
ml/scripts/create_label_index.py        one-time, already run
ml/scripts/analyse_truncation.py        diagnostic, optional
ml/scripts/run_overnight_experiments.py legacy; superseded by auto_experiment.py
ml/scripts/promote_model.py             MLflow registry promotion (M3-shared)
ml/scripts/compute_drift_baseline.py    M3 drift baseline
ml/data_extraction/ast_extractor.py     OFFLINE-ONLY; do not run in v4
ml/data_extraction/tokenizer.py         OFFLINE-ONLY; do not run in v4
```

### 6.6 Tests

```
ml/tests/test_api.py
ml/tests/test_cache.py
ml/tests/test_dataset.py
ml/tests/test_drift_detector.py
ml/tests/test_fusion_layer.py
ml/tests/test_gnn_encoder.py
ml/tests/test_model.py
ml/tests/test_preprocessing.py
ml/tests/test_promote_model.py
ml/tests/test_trainer.py
```

All must remain green during v4. Add: `test_auto_experiment.py` once
the script lands (parser + pre-flight; not the training itself).

---

## 7. Assumptions

These are the implicit beliefs the plan rests on. If any breaks, the plan
needs revisiting.

| # | Assumption | If false |
|---|---|---|
| A1 | RTX 3070 8 GB laptop has ≥ 7 GB free at session start | Pre-flight fails fast; cannot run anything |
| A2 | `val_indices.npy` is byte-identical to v3 | F1 numbers are not comparable to 0.5069 — any "win" is meaningless |
| A3 | Graphs in `ml/data/graphs/` are unchanged since v3 | Same as A2 |
| A4 | LoRA r=16 + batch 16 fits in 8 GB with AMP | Fall back to r=8 + batch 8 |
| A5 | The plateau at v3 is hyperparameter-bound, not data-bound | If data-bound, no hyperparameter search can break 0.5069 — need a v5 plan with new data or arch (out of scope) |
| A6 | The 10 % stratified subsample is a faithful proxy for full-data F1 | Calibrate empirically: first 5 confirms tell us the smoke→confirm correlation |
| A7 | Per-class floors hold uniformly (not just on average) | Macro-F1 win that breaks a floor is rejected — by design |
| A8 | MLflow tracking URI is reachable for the entire session | If not, runs are not logged; the operator may be tempted to keep them — don't |
| A9 | Disk has room for ~12 checkpoints (~5 GB each) during a session | Old non-best checkpoints should be cleaned by `auto_experiment.py` after each branch reset |

---

## 8. Expectations (the contract)

### 8.1 Per smoke run
- **Time:** 3–8 minutes
- **VRAM peak:** < 7.5 GB
- **MLflow artifact:** one run under `sentinel-retrain-v4`, tagged `regime=smoke`
- **Stdout last lines:** `SENTINEL_SCORE=…`, `PEAK_VRAM_MB=…`, `REGIME=smoke`

### 8.2 Per confirm run
- **Time:** 30–60 minutes
- **VRAM peak:** < 7.5 GB
- **MLflow artifact:** one run, tagged `regime=confirm`, with full per-class F1 columns
- **Stdout last lines:** `SENTINEL_SCORE=…`, `PEAK_VRAM_MB=…`, `REGIME=confirm`

### 8.3 Per session
- **Duration:** 8–12 hours unattended overnight
- **Output:** ~50–100 smoke runs, ~5–12 confirm runs, ≤ 1 winner
- **Files updated:** `results.tsv`, MLflow runs, possibly one new
  `multilabel-v4-*` checkpoint family
- **Files NOT updated:** anything in `ml/data/splits/`, `ml/data/graphs/`,
  `ml/src/models/`, `ml/src/preprocessing/` (hash guard enforces)

---

## 9. Predictions (honest forecast)

| # | Prediction | Probability | Mitigation |
|---|---|---|---|
| P1 | First session does not produce a winner above 0.5069 | High (60–70 %) | Plan calls for 1–3 nights; per-class floors prevent false wins |
| P2 | At least one OOM occurs early (batch×r combination) | Very high (90 %) | Exit code 2 → autoresearch loop discards and tries another knob |
| P3 | Smoke→confirm correlation is weaker than expected (smoke says win, confirm says not) | Medium (50 %) | First 5 confirms calibrate the smoke threshold; raise it if needed |
| P4 | DoS class breaks its floor in a few runs (it's small) | Medium (40 %) | Per-class floor check kicks in; agent reverts |
| P5 | A v4 winner emerges with focal γ ∈ [1.5, 2.5], LoRA r=16, DoS sampler ON | Best guess: 60 % conditional on plan running | None needed if it happens |
| P6 | Plateau holds; no v4 winner across 2 sessions | 25 % | Re-evaluate v5 path: weighted sampler families, per-class focal α, possibly k-fold thresholds |
| P7 | Wall-clock variance from thermal throttling on a laptop confuses the time budget | Medium (50 %) | Track wall-clock in `results.tsv`; flag outliers; stop session if average smoke time creeps > 12 min |
| P8 | An audit hardening item turns out to be already fixed (e.g. #15 docstring) | Low (20 %) | Quick verification before opening a fix PR |
| P9 | Inference key rename `threshold → thresholds` (Fix #6) breaks an undocumented downstream consumer | Medium (30 %) | M4 nodes already updated; verify M6 Phase 1 reads `thresholds` |
| P10 | A "free" win shows up: simply enabling the DoS weighted sampler without other changes lifts F1 by 0.005–0.015 | Medium-high (55 %) | Worth running first — single-knob baseline |

---

## 10. Acceptance / Success Criteria

### 10.1 v4 retrain (the headline goal)

A v4 checkpoint passes if **all** of the following hold:

- [ ] `tune_threshold.py` reports tuned F1-macro **> 0.5069** on `val_indices.npy`
- [ ] No per-class F1 below the v4 floor in §5.2
- [ ] Same `val_indices.npy` (sha256 match) as v3
- [ ] Architecture string in checkpoint config = `"cross_attention_lora"`
- [ ] Locked files unchanged (hash guard passed every run in the session)
- [ ] MLflow run logged in experiment `sentinel-retrain-v4` with all knobs as params and per-class F1 as metrics
- [ ] Checkpoint loaded by `predictor.py` cleanly in CI (test exists in `ml/tests/test_api.py`)
- [ ] `multilabel-v4-<runname>_best_thresholds.json` written and parseable
- [ ] `promote_model.py --stage Staging --val-f1-macro <new>` succeeds

If macro-F1 is up but a per-class floor breaks → **rejected**, even if
nominally better than 0.5069. This is the rule, not a guideline.

### 10.2 Autoresearch harness

- [ ] `auto_experiment.py --regime smoke` completes in < 8 min on the
      target laptop (3070 8 GB)
- [ ] Same script `--regime confirm` reproduces v3 within ±0.01 F1 on
      v3 hyperparameters (sanity)
- [ ] Locked-files hash guard rejects an intentionally-modified
      `sentinel_model.py` (test must pass)
- [ ] `program.md` enumerates the search space and forbidden edits
- [ ] Operator can Ctrl-C and the latest committed branch is the latest
      candidate (no orphaned partial checkpoints)
- [ ] `results.tsv` is well-formed TSV that loads cleanly into pandas

### 10.3 Audit hardening (background; not blocking)

- [ ] Audit #9: `process_source()` cleans temp files even on SIGTERM
- [ ] Audit #10/#12: hashing strategy unified; `hash_utils.py` either
      complete or removed
- [ ] Audit #14: `SENTINEL_TRACE=1` env flag implemented
- [ ] Audit #15: `FocalLoss` docstring corrected
- [ ] Audit #16: `CLASS_NAMES` extracted to `ml/src/preprocessing/labels.py`
- [ ] Audit #17: `peft` import deferred so pytest is quiet
- [ ] Audit #18: documented why LoRA still uses pickle; safetensors
      explored for non-LoRA tensors

---

## 11. Anything around it

### 11.1 What v4 is NOT trying to do

- Change the architecture (frozen — would invalidate M2 keys + M5
  verifier, see ADR-007 / ADR-025)
- Add new vulnerability classes (NUM_CLASSES is append-only, but
  appending requires retraining from scratch)
- Use new data (BCCC-SCsVul-2024 freeze; Move 9 multi-contract is
  post-M6)
- Replace CodeBERT (qwen2.5-coder-7b is interesting but blows the VRAM
  budget on this laptop)
- Train without LoRA (full fine-tune doesn't fit in 8 GB)
- Swap GAT for a different GNN (changes graph schema or fusion shape)

### 11.2 What v4 *enables* downstream

- M2 distillation: ProxyMLP(128→64→32→10) trains against the v4 teacher
  → new `proving_key.pk` + `verification_key.vk` → new
  `ZKMLVerifier.sol` → redeploy on Sepolia
- M5: `submitAudit` calls hit the new verifier address; old proofs from
  v3 teacher are invalid (acceptable; it's a model upgrade)
- M3: drift baseline can use v4 val set as the "in-distribution"
  reference once M6 starts producing real warm-up traffic
- M6: `predict()` returns v4 scores; `thresholds` list is per-class as
  always; nothing in the API contract changes

### 11.3 The single most important rule

> Per-class floor breach **always** rejects the candidate, even if
> macro-F1 is up.

Why: SENTINEL is a security tool. A v4 that is 0.02 better on macro-F1
because it stopped detecting DoS is a regression in the only thing
DoS-affected users care about. The audit chain (M5 on-chain, M2 ZK)
inherits whatever the model claims; promoting a class-regressing model
makes that audit cryptographically guaranteed but factually weaker.

### 11.4 What "M1 done" looks like

- v3 superseded by v4 in `docs/STATUS.md`
- v4 thresholds JSON committed alongside checkpoint
- All audit hardening items from §10.3 closed
- `auto_experiment.py` and `program.md` documented and used at least
  once in production
- `validate_graph_dataset.py` still exits 0
- Predictor warm-up + threshold load tested in CI
- Cross-doc updates from autoresearch plan §11 applied

After that, M1 is "done for this dataset" — further gains require
either new data (Move 9 helps marginally; new corpus helps a lot) or a
new architecture (which restarts the entire ZKML cascade).
