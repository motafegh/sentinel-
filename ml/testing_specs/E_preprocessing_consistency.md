# E — Preprocessing Consistency

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.

---

## When This File Applies

Load this spec when:
- Launching a new training run (load alongside `F_new_run_checklist.md`)
- Changing the graph schema (load alongside `J_schema_migration.md`)
- Running inference on a new dataset and suspecting metric corruption
- Any time preprocessing settings in training or inference have changed

---

## E.1 — Mandatory Reading Order Before This Procedure

Before checking consistency, read these files in order:

1. `ml/src/data_extraction/windowed_tokenizer.py`
   - Source of truth for training tokenization: model name, window size, stride,
     max windows, comment stripping behaviour
2. `ml/scripts/retokenize_windowed.py`
   - Batch orchestration for offline token generation; reads `windowed_tokenizer.py`
3. `ml/src/inference/preprocess.py` — `ContractPreprocessor` class
   - Source of truth for inference-time tokenization
4. `ml/src/preprocessing/graph_schema.py`
   - Source of truth for graph feature dimensions and schema version

Do not proceed from memory. Read each file header and the relevant constants
before performing any comparison.

---

## E.2 — Tokenizer Path Alignment

Confirm the following four values match exactly between the offline training
pipeline and online inference. Read both sources before comparing.

| Parameter | Training source | Inference source |
|---|---|---|
| Tokenizer model name | `windowed_tokenizer.TOKENIZER_MODEL` | `ContractPreprocessor.TOKENIZER_NAME` |
| Window size (seq len) | `windowed_tokenizer.WINDOW_SIZE` | `ContractPreprocessor.MAX_TOKEN_LENGTH` |
| Stride | `windowed_tokenizer.STRIDE` | `_tokenize_sliding_window(stride=...)` |
| Max windows | `windowed_tokenizer.MAX_WINDOWS` | `process_source_windowed(max_windows=...)` |

**Known structural difference — read before interpreting:**

The standard `process_source()` path in `preprocess.py` produces a
**single-window** `[1, 512]` tensor. The offline training pipeline produces
`[MAX_WINDOWS, 512]` tensors. This is a documented intentional asymmetry
for the non-windowed inference path.

For windowed inference (`process_source_windowed()`), the sliding-window logic
must match the offline `retokenize_windowed.py` linspace subsampling exactly.
Read `_tokenize_sliding_window()` in `preprocess.py` and `_select_windows()` in
`windowed_tokenizer.py` side-by-side to confirm the subsampling logic is identical.

If the model was trained on windowed `[4, 512]` tokens, inference must use
`process_source_windowed()` — not `process_source()`. Confirm which path
`predictor.py` calls before drawing any metric conclusions.

---

## E.3 — Comment Stripping Alignment

Read the `strip_comments` parameter in both paths:

- Training: `windowed_tokenizer.tokenize_windowed_contract(strip_comments=True)`
  is the default — read the docstring to confirm whether the caller overrides it
- Inference: `preprocess.py` `_tokenize()` does NOT call `_strip_comments()`
  on the source before the single-window tokenize call

If comment stripping is applied during training but not during inference,
the token budget allocation differs — confirm this is intentional and
documented. If it is unintentional, it is a preprocessing mismatch finding.

Write a gate assertion for comment stripping alignment status.

---

## E.4 — Graph Schema Alignment

Confirm the `FEATURE_SCHEMA_VERSION` and `NODE_FEATURE_DIM` used at inference
match what the model was trained on.

1. Read `ml/src/preprocessing/graph_schema.py` — note the current
   `FEATURE_SCHEMA_VERSION` and `NODE_FEATURE_DIM`
2. Read the checkpoint metadata (path from `MEMORY.md`) — confirm which schema
   version the checkpoint was built with
3. Read `preprocess.py` — confirm it imports from `graph_schema.FEATURE_SCHEMA_VERSION`
   and that the cache key includes the schema version (it should; verify)

`preprocess.py` includes this in the cache key:
`contract_hash = f"{content_hash}_{FEATURE_SCHEMA_VERSION}"`

This means a schema version change automatically invalidates the inference cache.
Confirm this is working correctly if the schema has changed since the last run.

---

## E.5 — Re-tokenization Trigger Conditions

Run `ml/scripts/retokenize_windowed.py` when **any** of the following change:

- Tokenizer model name changes
- `WINDOW_SIZE`, `STRIDE`, or `MAX_WINDOWS` constants change
- `strip_comments` default changes
- The `--relabel-timestamp` flag behaviour changes

**Destructive side-effect warning:** `retokenize_windowed.py` overwrites
existing `.pt` token files in `ml/data/tokens_windowed/`. There is no
automatic backup. Archive or DVC-snapshot the existing token files
before running if they may be needed for reproducibility.

After re-tokenization, the token cache used by the DataLoader must also be
invalidated. Read `ml/src/datasets/dual_path_dataset.py` to confirm the cache
key format and which files must be deleted or regenerated.

---

## E.6 — Re-extraction Trigger Conditions

Run `ml/scripts/reextract_graphs.py` when **any** of the following change:

- `FEATURE_SCHEMA_VERSION` advances in `graph_schema.py`
- `NODE_FEATURE_DIM` or any node feature encoding logic changes
- New edge types are added or existing edge type indices change
- New node types are added
- Slither version changes (different CFG/AST output)

**Destructive side-effect warning:** `reextract_graphs.py` overwrites
existing `.pt` graph files in `ml/data/graphs/`. Archive or DVC-snapshot
before running.

After re-extraction:
- The smoke suite (Section D) must be re-run — schema constants in
  `_common.py` may need updating if the schema version advanced
- The DataLoader cache must be invalidated (same as re-tokenization)
- Run the schema-dim gate test before any training run

---

## E.7 — DataLoader Cache Validation

Read `ml/src/datasets/dual_path_dataset.py` before verifying the DataLoader cache:

- Confirm the cache key format (what fields constitute a cache hit)
- Confirm the cache version recorded in the current split files matches
  the preprocessing settings you have confirmed above
- If `retokenize_windowed.py` was run with a new `--relabel-timestamp`,
  confirm the split cache references the new timestamp, not the old one

The v10 `--relabel-timestamp` omission is a documented historical failure
mode: the DataLoader silently loaded stale pre-relabel token files because
the cache key still pointed to the old timestamp. Verify the timestamp is
current before proceeding to training.

---

## E.8 — Completion Attestation

After completing this section, append to the relevant run pre-flight doc:

```
## Procedure Attestation — E_preprocessing_consistency — <ISO date>
Steps completed:   E.1 sources read: PASS/FAIL
                   E.2 tokenizer alignment: PASS/FAIL/UNVERIFIED
                   E.3 comment stripping alignment: PASS/FAIL/UNVERIFIED
                   E.4 schema alignment: PASS/FAIL/UNVERIFIED
                   E.5 re-tokenization required: YES/NO (reason if YES)
                   E.6 re-extraction required: YES/NO (reason if YES)
                   E.7 DataLoader cache: PASS/FAIL/UNVERIFIED
Steps skipped:     [any skipped steps + explicit reason]
Unverified items:  [anything not confirmable from source files]
New findings:      [link to audit doc entry, or "none"]
Written to:        [path of this attestation]
```
