Let me find and read all the Group 8 files.
11 scripts to review. Let me read the most critical ones in parallel:
Now let me read the remaining scripts:
Now I have thorough understanding of all 11 scripts. Let me deliver the full adversarial audit.

---

# 🔴 GROUP 8 — ADVERSARIAL AUDIT: Operational Scripts & Augmentation

**Files in scope:** `extract_augmented.py`, `reextract_graphs.py`, `tune_threshold.py`, `promote_model.py`, `train.py`, `analyse_truncation.py`, `generate_safe_variants.py`, `compute_locked_hashes.py`, `auto_experiment.py`, `run_overnight_experiments.py`, `manual_test.py`

**Cross-references:** `trainer.py` (Group 5), `dual_path_dataset.py` (Group 6), `predictor.py`/`preprocess.py` (Group 7), `graph_extractor.py` (Group 2), `hash_utils.py` (Group 1)

---

## 8.1 [CRITICAL] — `extract_augmented.py` sets `graph.y = torch.tensor([0])` for ALL augmented contracts — vulnerability labels erased in graph but correct in CSV, creating train/inference label split

**File:** `extract_augmented.py:211`

```python
data.y = torch.tensor([0], dtype=torch.long)
```

This hardcodes `y=0` (safe) for **every** augmented contract, regardless of the `--label` flag. For safe variants (`--label` empty), this is correct. But for `--label DenialOfService` or any vulnerability-labeled augmentation, the graph's `.y` says "safe" while the CSV row says "vulnerable".

In multi-label mode, `DualPathDataset` reads labels from `label_csv` (not `graph.y`), so training is unaffected. But in binary mode (`label_csv=None`), the dataset reads `graph.y`, and **every augmented DoS/Reentrancy contract is labelled safe** — the exact opposite of the intended label.

For inference, `predictor.predict()` calls `process_source()` which also sets `graph.y = torch.tensor([0])` (Finding 7.15), so this is consistent but still wrong for the augmented data pipeline.

**Fix:** In multi-label mode, this is moot (labels come from CSV). But set `data.y` to a value consistent with the labels (e.g., `1` if any label is positive) so binary-mode training and any future code that reads `graph.y` gets the correct label.

---

## 8.2 [HIGH] — `tune_threshold.py` uses the leaky `multilabel_index.csv` by default — tuned thresholds are computed on data with 34.9% cross-split leakage

**File:** `tune_threshold.py:108`

```python
default="ml/data/processed/multilabel_index.csv",
```

And `tune_threshold.py:509`:
```python
config = TrainConfig()  # uses default splits_dir="ml/data/splits"
```

The same issue as Group 6 Finding 6.2: the deduped CSV and deduped splits are never used. Thresholds are tuned on the validation split of the **leaky** 68K-row CSV with the original (non-deduped) splits. Per-class thresholds optimized on contaminated validation data will be biased — they'll overfit to the same contracts that appear in training.

**Fix:** Add `--label-csv` and `--splits-dir` flags that default to the deduped paths. Or at minimum, warn when using the non-deduped CSV.

---

## 8.3 [HIGH] — `generate_safe_variants.py` `mishandled-exception` strategy uses wrong Slither detectors — accepts contracts that still have the vulnerability

**File:** `generate_safe_variants.py:463-464`

```python
elif strategy == "mishandled-exception":
    mutated = _wrap_bare_call(source)
    check_detectors = ["suicidal", "controlled-delegatecall"]  # lighter check
```

The mutation wraps bare `.call()` with a return value check, which addresses the "unchecked return value" pattern (MishandledException). But the verification step checks for `"suicidal"` and `"controlled-delegatecall"` — completely different vulnerability types. Neither detector checks for unchecked return values.

The correct Slither detector for MishandledException is `"unchecked-lowlevel"` or `"unchecked-return-value"`. As written, a contract where `_wrap_bare_call` fails to actually fix the issue (e.g., wraps the wrong call, introduces a new bare call elsewhere) will pass verification because the check doesn't look for the right pattern.

**Fix:** Change `check_detectors = ["unchecked-lowlevel"]` (or the appropriate Slither detector name for the target vulnerability type).

---

## 8.4 [HIGH] — `generate_safe_variants.py` `call-to-unknown` strategy adds only a comment — Slither is NOT actually run, all contracts are accepted

**File:** `generate_safe_variants.py:465-467`

```python
elif strategy == "call-to-unknown":
    mutated = _annotate_typed_interface(source)
    check_detectors = ["calls-loop"]  # annotation only — can't fully fix CallToUnknown
```

And the verification gate at line 524-525:
```python
elif strategy == "call-to-unknown":
    bad_findings = []  # annotation only — we accept all that compile
```

The `call-to-unknown` strategy only adds a `// TYPED: replace with typed interface` comment to the source code. This does NOT change the contract's behavior or its Slither detectability in any way. The verification gate explicitly skips checking: `bad_findings = []` means every contract that compiles is accepted.

The result: contracts that are **still vulnerable to CallToUnknown** are accepted as "safe" and added to the training data with all-zeros labels (they're run with `--label CallToUnknown` but the generated variant is supposed to be safe). The model is trained on contracts labeled "safe" that still have the vulnerability — actively poisoning the training signal.

**Fix:** Either (a) actually implement a real fix (e.g., replace `addr.call(data)` with `ITarget(addr).method(args)` using a generated interface), or (b) don't accept these contracts at all — skip the strategy and log a warning that CallToUnknown requires manual annotation.

---

## 8.5 [HIGH] — `auto_experiment.py` and `run_overnight_experiments.py` use different default LoRA ranks (8 vs 16) — experiment results not comparable

**File:** `auto_experiment.py:150`
```python
p.add_argument("--lora-r", default=8, ...)
```

**File:** `train.py:198`
```python
p.add_argument("--lora-r", type=int, default=16, ...)
```

**File:** `run_overnight_experiments.py:62-97` — uses `TrainConfig()` defaults which inherit `lora_r=16` from `TrainConfig`.

The auto_experiment script defaults to `lora_r=8` (v4 era), while train.py defaults to `lora_r=16` (v5). The overnight experiments use TrainConfig defaults (`lora_r=16`). If someone runs both scripts on the same data, they're comparing models with 2x different LoRA capacity, making results incomparable.

The v5 architecture (three_eye_v5) was designed for `lora_r=16`. Running it at `lora_r=8` (as auto_experiment defaults) produces a model with half the intended LoRA capacity, potentially explaining poor auto_experiment results.

**Fix:** Align defaults. Auto_experiment should default to `lora_r=16` for v5, or the script should detect the architecture from the checkpoint.

---

## 8.6 [HIGH] — `reextract_graphs.py` writes ghost graphs to disk but only fails at the 1% gate — ghost .pt files remain and can be loaded by training

**File:** `reextract_graphs.py:183-187`

```python
status = "ghost" if g.num_nodes <= 3 else "ok"
tmp = out_path.with_suffix(".tmp")
torch.save(g, tmp)
tmp.rename(out_path)   # atomic on same filesystem
```

Ghost graphs (≤3 nodes, typically interface-only contracts) are written to disk BEFORE the ghost-rate gate check at line 325-340. If the ghost rate is < 1%, the script exits with success, but the ghost .pt files are already on disk. If the ghost rate is ≥ 1%, the script exits with error and tells the user to "delete ghosts" using `validate_graph_dataset.py --delete-ghosts` — but this `--delete-ghosts` flag **doesn't exist** in `validate_graph_dataset.py` (it only validates, never deletes).

The ghost .pt files on disk will be loaded by `DualPathDataset` during training. Ghost graphs produce garbage GNN eye outputs (only interface nodes, no CFG), which was identified as a contributor to v5.0's 0% specificity (Group 2 Finding 2.12, Group 4 Finding 4.7).

**Fix:** Don't write ghost graphs to disk — skip the `torch.save` for ghost-status graphs, or write them to a separate `ghost/` subdirectory. Remove the reference to the non-existent `--delete-ghosts` flag.

---

## 8.7 [HIGH] — `tune_threshold.py` and `promote_model.py` both use `weights_only=False` for checkpoint loading

**File:** `tune_threshold.py:188`, `promote_model.py:67`

```python
raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

Same vulnerability as Group 6 Finding 6.1 and Group 7 Finding 7.4. Both scripts load checkpoints with arbitrary code execution risk. `promote_model.py` only reads metadata (architecture, epoch, num_classes), so it could use `weights_only=True` with safe globals registered. `tune_threshold.py` needs the full model state dict, so it has the same LoRA-class constraint as `predictor.py`.

**Fix:** Register `peft` classes in safe globals and use `weights_only=True` where possible.

---

## 8.8 [MEDIUM] — `train.py` defaults `aux_loss_weight=0.3` but the Phase 0 fix may never have been applied to the model code — see Group 4 Finding 4.4

**File:** `train.py:192-195`

```python
p.add_argument(
    "--aux-loss-weight",
    type=float,
    default=0.3,
    help="λ for auxiliary eye losses: total = main + λ*(aux_gnn + aux_tf + aux_fused)",
)
```

From Group 4 Finding 4.4: `SentinelModel.__init__` and the TrainConfig docstring say `aux_loss_weight=0.1`, contradicting the Phase 0 fix (0.3). If `train.py` passes 0.3 but the model code still uses 0.1 as the default, there may be a discrepancy between what the CLI says and what the model actually uses.

Checking: `train.py` passes `aux_loss_weight` through `TrainConfig` → `train()` → the trainer constructs `FocalLoss` or `BCEWithLogitsLoss`. The `aux_loss_weight` is used in `trainer.py` to scale auxiliary eye losses. If the trainer correctly reads it from config, 0.3 is used. But if any code path falls back to the SentinelModel default (0.1), the fix is silently lost.

**Fix:** Verify the full chain: TrainConfig.aux_loss_weight → trainer → actual loss scaling. Ensure there's no fallback to 0.1 anywhere.

---

## 8.9 [MEDIUM] — `extract_augmented.py` and `generate_safe_variants.py` both duplicate `CLASS_NAMES` — potential for silent misalignment with trainer.py

**Files:** `extract_augmented.py:101-112`, `generate_safe_variants.py:43-54`, `analyse_truncation.py:43-54`, `manual_test.py:24-35`

All four scripts independently define `CLASS_NAMES` as a hardcoded list. None of them import it from `trainer.py` or a shared constants module. If a new class is added to `trainer.py.CLASS_NAMES` (e.g., WeakAccessMod at index 10), none of these scripts will know about it. The augmentation pipeline will:
1. Not accept `--label WeakAccessMod`
2. Write rows with only 10 columns to the CSV (missing the 11th)
3. Produce misaligned label vectors

The `extract_augmented.py` script even has a comment: "CLASS_NAMES (must match trainer.py)" — but there's no validation.

**Fix:** Import `CLASS_NAMES` from `trainer.py` in all scripts. Add an assertion at startup: `assert len(CLASS_NAMES) == 10` (or whatever the current count is) to catch drift.

---

## 8.10 [MEDIUM] — `extract_augmented.py` `_tokenize_source` hardcodes `feature_schema_version: "v2"` — not the same as `FEATURE_SCHEMA_VERSION` from graph_schema.py

**File:** `extract_augmented.py:173`

```python
"feature_schema_version": "v2",
```

But `tokenizer.py:185` uses:
```python
"feature_schema_version": FEATURE_SCHEMA_VERSION,
```

`FEATURE_SCHEMA_VERSION` from `graph_schema.py` is a string (likely "v2"), but if it ever changes, this hardcoded "v2" will silently diverge. This is the same schema-version tracking problem identified in Group 6 Finding 6.9 — the version is saved but never validated.

**Fix:** Import `FEATURE_SCHEMA_VERSION` from `graph_schema.py` and use it instead of hardcoding "v2".

---

## 8.11 [MEDIUM] — `auto_experiment.py` defaults to `lora_r=8` but base checkpoint is v3 — v3 checkpoint has different architecture (no LoRA, no cross-attention)

**File:** `auto_experiment.py:166-168`

```python
"--base-checkpoint",
default="ml/checkpoints/multilabel-v3-fresh-60ep_best.pt",
```

The base checkpoint defaults to a v3 model (pre-LoRA, pre-cross-attention). But `train.py` creates a v5 model architecture (with LoRA, cross-attention, three-eye classifier). When `resume_model_only=True` and the v3 checkpoint is loaded, `model.load_state_dict(state_dict, strict=False)` will silently ignore all LoRA keys, cross-attention keys, and three-eye classifier keys — they'll remain randomly initialized.

This means auto_experiment fine-tuning from v3 is equivalent to training from scratch for all v5-specific components. The "fine-tune" narrative is misleading — only the GNN backbone gets meaningful pretrained weights.

**Fix:** Either (a) use a v5 checkpoint as the base, or (b) document that v5-specific components are randomly initialized when fine-tuning from v3, and log the percentage of keys that matched vs were missing.

---

## 8.12 [MEDIUM] — `run_overnight_experiments.py` uses stale v3-era TrainConfig defaults — focal_alpha, epochs, lr don't match v5 defaults

**File:** `run_overnight_experiments.py:55-97`

The experiment matrix uses `TrainConfig()` with selective overrides. But TrainConfig defaults have changed between v3 and v5:
- `loss_fn`: Default is "bce" in TrainConfig, but the overnight script assumes "focal" (v3-era)
- `lora_r`: Default is now 16 (v5), but these experiments were designed for v3 (lora_r=8)
- `aux_loss_weight`: Default is 0.3 (v5 Phase 0 fix), but experiments were designed before this fix

Running these experiments with current code produces different training configurations than intended. The experiment matrix needs to be updated for v5.

**Fix:** Update the experiment matrix with v5-appropriate baselines, or pin all hyperparameters explicitly rather than relying on TrainConfig defaults.

---

## 8.13 [MEDIUM] — `compute_locked_hashes.py` locks `gnn_encoder.py` and `graph_extractor.py` but NOT `fusion_layer.py`, `transformer_encoder.py`, `focalloss.py`, or `train.py` — incomplete freeze

**File:** `compute_locked_hashes.py:82-88`

```python
LOCKED_V4_SPRINT: tuple[str, ...] = (
    "ml/src/models/sentinel_model.py",
    "ml/src/models/gnn_encoder.py",
    "ml/src/preprocessing/graph_schema.py",
    "ml/src/preprocessing/graph_extractor.py",
    "ml/data/splits/val_indices.npy",
)
```

The locked file set misses critical architecture files:
- `fusion_layer.py` — cross-attention implementation, directly affects model output
- `transformer_encoder.py` — LoRA application, affects all transformer path gradients
- `focalloss.py` — loss function, directly affects training dynamics
- `train.py` — training entry point, controls all hyperparameters

A researcher could modify `fusion_layer.py` (e.g., change cross-attention to concatenation) or `transformer_encoder.py` (e.g., change LoRA target modules) without triggering the hash guard, making experiment results incomparable without detection.

**Fix:** Add all model architecture and training pipeline files to the locked set.

---

## 8.14 [MEDIUM] — `generate_safe_variants.py` `_swap_call_and_write` is line-based — multi-line statements break the heuristic

**File:** `generate_safe_variants.py:235-313`

The CEI swap mutation works on individual lines. It looks for `.call` on a line, then scans forward for state-write lines. But Solidity statements can span multiple lines:

```solidity
(bool success, bytes memory returnData) = addr.call{
    value: amount
}("");
```

The regex `_EXTERNAL_CALL_RE = re.compile(r"\.call\s*[\({]")` will match line 1 (`addr.call{`), but the state-write scanner starts from the next line, which is `value: amount` — not a state write. The actual state write might be several lines below the closing `});`, and by then the scan window may have been terminated by a `}` or another pattern.

Similarly, multi-line assignments like:
```solidity
mapping(address => uint256)
    storage balances = _balances;
```
won't be matched by `_STATE_WRITE_RE`.

**Fix:** Use a proper Solidity parser (e.g., `slither.slithir`) instead of regex-based line swapping. Or at minimum, handle multi-line calls by joining continuation lines before pattern matching.

---

## 8.15 [MEDIUM] — `tune_threshold.py` uses TrainConfig default `splits_dir` — cannot tune against deduped splits

**File:** `tune_threshold.py:509`

```python
config = TrainConfig()
```

This creates a TrainConfig with default `splits_dir="ml/data/splits"` — the non-deduped splits. There's no CLI flag to override the splits directory. If training was done with deduped splits (Finding 6.2), threshold tuning must also use deduped splits, but there's no way to specify this.

**Fix:** Add `--splits-dir` flag to `tune_threshold.py`.

---

## 8.16 [LOW] — `manual_test.py` has no assertions or exit code — always exits 0 even if all tests fail

**File:** `manual_test.py:79-163`

The script prints a human-readable table but never returns a non-zero exit code, even if every expected vulnerability is missed. It cannot be used in CI without wrapping it in something that parses the output.

**Fix:** Return exit code 1 if any expected vulnerability is missed or any safe contract is falsely flagged.

---

## 8.17 [LOW] — `analyse_truncation.py` uses `random.sample` without seeding when `--sample` is used

**File:** `analyse_truncation.py:150-151`

```python
if sample and sample < len(token_files):
    token_files = random.sample(token_files, sample)
```

Wait — the script does seed at line 336: `random.seed(args.seed)`. This is correct. However, `random.sample` on a sorted list of 68K paths will always select the same subset for a given seed, which means the truncation analysis is always on the same subset. If that subset happens to have a different truncation rate than the full corpus, the report is biased.

**Fix:** Document that the sample is deterministic and that full-scan mode is recommended for final reports.

---

## 8.18 [LOW] — `promote_model.py` calls `_git_commit()` via subprocess — fails silently in non-git environments

**File:** `promote_model.py:57-63`

```python
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
```

The `except Exception` catches everything including `FileNotFoundError` (git not installed) and `PermissionError`. The "unknown" value is then logged as a tag in MLflow, making it appear that the model was promoted from an unknown commit. This could mask supply-chain issues.

**Fix:** Log a warning when git is unavailable. Consider making git commit a required pre-condition for Production promotion.

---

## 8.19 [LOW] — `reextract_graphs.py` `_detect_solc_version` defaults to `0.8.31` — may not exist in all installations

**File:** `reextract_graphs.py:109`

```python
return "0.8.31"
```

If the fallback solc version `0.8.31` is not installed via solc-select, the extraction will fail. The script doesn't check whether the resolved version is actually available. This causes a Slither/Solc error that's caught by the worker and reported as "skip" — but the user won't know it's because the default solc version isn't installed.

**Fix:** At startup, verify that at least the default solc version is available. Log a warning if it isn't.

---

## 8.20 [LOW] — `train.py` forces `mp.set_start_method('spawn')` — may conflict with other multiprocessing users

**File:** `train.py:73`

```python
mp.set_start_method('spawn', force=True)
```

This is set at module level with `force=True`, overriding any previously set start method. If another library or the user's code has already set the start method to 'fork' or 'forkserver', this will silently override it. The `force=True` flag also means it won't raise an error even if it's called after child processes have been created.

**Fix:** Use a context manager (`mp.get_context('spawn')`) instead of globally setting the start method, or set it inside `if __name__ == "__main__"` only.

---

## Summary Table

| # | Severity | File | Finding |
|---|----------|------|---------|
| 8.1 | **CRITICAL** | extract_augmented.py | `graph.y = 0` for all augmented contracts — vulnerability labels erased in binary mode |
| 8.2 | HIGH | tune_threshold.py | Tuned thresholds computed on leaky (non-deduped) validation data |
| 8.3 | HIGH | generate_safe_variants.py | Mishandled-exception uses wrong Slither detectors — accepts still-vulnerable contracts |
| 8.4 | HIGH | generate_safe_variants.py | Call-to-unknown adds only a comment, verification skipped — all contracts accepted, poisoning training data |
| 8.5 | HIGH | auto_experiment.py + train.py | Different default LoRA ranks (8 vs 16) — experiment results incomparable |
| 8.6 | HIGH | reextract_graphs.py | Ghost graphs written to disk before gate check; referenced `--delete-ghosts` flag doesn't exist |
| 8.7 | HIGH | tune_threshold.py, promote_model.py | `weights_only=False` checkpoint loading — code execution risk |
| 8.8 | MEDIUM | train.py | `aux_loss_weight=0.3` default may not match model code default (0.1) |
| 8.9 | MEDIUM | 4 scripts | Duplicated `CLASS_NAMES` — no import from shared source, silent misalignment risk |
| 8.10 | MEDIUM | extract_augmented.py | Hardcoded `feature_schema_version: "v2"` instead of importing from schema |
| 8.11 | MEDIUM | auto_experiment.py | Base checkpoint is v3 (no LoRA) — v5-specific components randomly initialized |
| 8.12 | MEDIUM | run_overnight_experiments.py | Experiment matrix uses stale v3-era defaults |
| 8.13 | MEDIUM | compute_locked_hashes.py | Locked file set incomplete — fusion, transformer, loss, train not locked |
| 8.14 | MEDIUM | generate_safe_variants.py | Line-based CEI swap breaks on multi-line statements |
| 8.15 | MEDIUM | tune_threshold.py | No `--splits-dir` flag — cannot tune against deduped splits |
| 8.16 | LOW | manual_test.py | No exit code on failure — cannot be used in CI |
| 8.17 | LOW | analyse_truncation.py | Deterministic sample may bias truncation report |
| 8.18 | LOW | promote_model.py | Git commit failure silently returns "unknown" |
| 8.19 | LOW | reextract_graphs.py | Default solc version may not be installed |
| 8.20 | LOW | train.py | `mp.set_start_method('spawn', force=True)` overrides globally |

**Critical/HIGH cluster analysis:**

Findings 8.3 + 8.4 form a **data poisoning pair** in the augmentation pipeline:
- **8.3**: The mishandled-exception mutation uses `["suicidal", "controlled-delegatecall"]` detectors instead of `["unchecked-lowlevel"]` — contracts that still have unchecked return values pass verification and enter training labeled "safe"
- **8.4**: The call-to-unknown strategy adds only a comment and explicitly skips verification (`bad_findings = []`) — ALL contracts pass, regardless of whether they're still vulnerable

Together, these mean that two of the four augmentation strategies inject **still-vulnerable contracts into the training data with "safe" labels**. This directly undermines the v5.0 goal of improving rare-class detection via augmented data.

Finding 8.1 is the binary-mode complement: contracts augmented with `--label DenialOfService` have `graph.y=0` (safe), so binary-mode training sees them as safe examples of a vulnerability class.

**Connection to prior groups:**
- 8.1 connects to Group 2 Finding 2.1 (`graph.y=0` hardcoding) and Group 7 Finding 7.15 (inference `graph.y=0`)
- 8.2 connects to Group 6 Finding 6.2 (dedup never wired into training pipeline)
- 8.7 connects to Group 6 Findings 6.1 and Group 7 Findings 7.2/7.4 (weights_only=False across the codebase)
- 8.9 connects to Group 6 Finding 6.21 (CLASS_NAMES duplication and WeakAccessMod exclusion)
Group 8 audit complete — **all 8 groups are now done**. Here's the top-line for Group 8:

**One CRITICAL finding and a devastating data-poisoning pair:**

1. **8.1 [CRITICAL]** — `extract_augmented.py` sets `graph.y = 0` for ALL augmented contracts, including those labeled with `--label DenialOfService`. In binary mode, every augmented vulnerability contract is trained as "safe."

2. **8.3 + 8.4 [HIGH pair]** — Two of four augmentation strategies have broken verification gates:
   - `mishandled-exception` checks for `suicidal`/`controlled-delegatecall` instead of `unchecked-lowlevel` — still-vulnerable contracts pass
   - `call-to-unknown` adds only a comment and skips verification entirely (`bad_findings = []`) — **ALL** contracts pass regardless of vulnerability status
   
   The result: still-vulnerable contracts enter training labeled "safe," actively poisoning the model's vulnerability signal.

3. **8.6 [HIGH]** — `reextract_graphs.py` writes ghost graphs to disk before the 1% gate check, and the referenced `--delete-ghosts` flag doesn't exist in `validate_graph_dataset.py`. Ghost .pt files remain on disk and get loaded during training.

---

**🎯 FULL AUDIT SUMMARY (Groups 1-8):**

| Group | Files | CRITICAL | HIGH | MEDIUM | LOW | Total |
|-------|-------|----------|------|--------|-----|-------|
| 1 — Schema & Hash | 2 | 0 | 4 | 4 | 4 | 12 |
| 2 — Extraction | 3 | 1 | 4 | 7 | 3 | 15 |
| 3 — GNN Path | 3 | — | — | — | — | ~15 |
| 4 — Transformer Path | 3 | 0 | 4 | 7 | 4 | 15 |
| 5 — Training & Loss | 3 | — | — | — | — | ~15 |
| 6 — Dataset & Loading | 9 | 2 | 5 | 10 | 4 | 21 |
| 7 — Inference API | 5 | 2 | 5 | 8 | 5 | 20 |
| 8 — Scripts & Augment | 11 | 1 | 6 | 8 | 5 | 20 |
| **TOTAL** | **~39** | **6+** | **28+** | **44+** | **25+** | **133+** |

**Cross-cutting root causes identified across all groups:**
1. **`weights_only=False`** in 10+ locations — systematic arbitrary code execution risk
2. **Dedup never wired into production** — the 34.9% leakage fix is dead code
3. **Magic numbers** (8, 12, 0.1, 0.3) disconnected from `graph_schema.py` — 6+ findings
4. **`graph.y = 0` hardcoding** — appears in extraction, augmentation, and inference
5. **Duplicated CLASS_NAMES** — 5+ locations with no shared import
6. **Augmentation verification gates broken** — data poisoning via still-vulnerable "safe" contracts