# J — Schema Migration

> Always load `00_rules.md` before following this procedure.
> Apply Rule 2 (gate assertions + completion attestation) at every step.
>
> **Last revised: 2026-06-14** (post-Run-12 launch). **Major corrections**:
>
> 1. **Canonical source moved** (Stage 7B seam swap, 2026-06-12). The REAL
>    `graph_schema.py` is at `data_module/sentinel_data/representation/graph_schema.py`.
>    The `ml/src/preprocessing/graph_schema.py` is now a 22-line shim that
>    re-exports the canonical. Always read the canonical source.
> 2. **`graph_extractor.py` is in `ml/src/preprocessing/`, NOT `ml/src/data_extraction/`**
>    (the old path in §J.2 was stale).
> 3. **`ml/scripts/{reextract_graphs,retokenize_windowed,validate_graph_dataset}.py` NO LONGER EXIST**
>    (verified via `ls ml/scripts/`: 2026-06-14). They were replaced by the
>    `data_module/sentinel_data/cli.py` orchestrator (`represent` and `validate`
>    subcommands). The seam-swap is INCOMPLETE for the underlying impl files
>    (`ml/src/preprocessing/graph_extractor.py` 2,056 lines; `ml/src/data_extraction/windowed_tokenizer.py` 175 lines).
> 4. **Active data dirs**: graphs and tokens are now under the v3 export
>    `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/`, not `ml/data/graphs/`.
> 5. **Slither assertion target**: `graph_schema.py` asserts Slither >= 0.9.3
>    (unchanged from v9). Currently installed: 0.11.5.

---

## When This File Applies

- Changing any constant in `ml/src/preprocessing/graph_schema.py`
- Adding, removing, or reordering a node feature dimension in `FEATURE_NAMES`
- Adding or renaming an entry in `NODE_TYPES`, `EDGE_TYPES`, or `VISIBILITY_MAP`
- Changing `_build_node_features()` or `_build_cfg_node_features()` logic in
  `graph_extractor.py`
- After a Slither version upgrade that changes IR node types or property names
- Any time `FEATURE_SCHEMA_VERSION` must be incremented

Always load alongside: `E_preprocessing_consistency.md` (re-extraction and
re-tokenization trigger conditions) and `F_new_run_checklist.md` (pre-launch
gates for the retrain after migration).

---

## J.1 — What Constitutes a Schema Change

A schema change is any modification whose effect reaches a training graph
`.pt` file, an inference cache entry, or a model weight dimension. Read
`graph_schema.py` `CHANGE POLICY` section before classifying.

**Canonical source**: `data_module/sentinel_data/representation/graph_schema.py`
(read this file, not the shim). The shim at `ml/src/preprocessing/graph_schema.py`
(22 lines) re-exports the constants for backward compat with `ml/` code paths
that haven't been seam-swapped.

| Change type | Schema change? | Re-extract? | Re-tokenize? | Retrain? |
|---|---|---|---|---|
| New node feature dimension (add to `FEATURE_NAMES` + `NODE_FEATURE_DIM`) | YES | YES | NO | YES |
| Remove or reorder node feature dimension | YES | YES | NO | YES |
| Change feature encoding logic (e.g. raw → log-normalized) | YES | YES | NO | YES |
| New node type entry in `NODE_TYPES` (append only) | YES | YES | NO | YES |
| Re-number existing `NODE_TYPES` entries | YES | YES | NO | YES |
| New edge type entry in `EDGE_TYPES` (append only) | YES | YES | NO | YES |
| Change `VISIBILITY_MAP` values | YES | YES | NO | YES |
| Change tokenizer window logic in `windowed_tokenizer.py` | NO | NO | YES | YES |
| Change `FEATURE_SCHEMA_VERSION` string only (no other change) | Version bump only | NO | NO | NO |
| Runtime-only edge type (e.g. `REVERSE_CONTAINS`=7 — never on disk) | NO | NO | NO | NO |
| Change `STRUCTURAL_PREFIX_TYPES` set | NO | NO | NO | YES (different prefix injection) |

**Rule: never insert in the middle of `NODE_TYPES` or `EDGE_TYPES`.** The IDs
must remain stable — inserting renumbers downstream entries and invalidates all
existing `.pt` files silently (no shape error, wrong meaning). Always append at
the end. If an ID must be retired, mark it as deprecated with a comment and
leave it in place.

---

## J.2 — Downstream Consumers

Before making any schema change, identify all files that reference the constant
being changed. Do not rely on memory — use code search:

```bash
# Find all files importing from graph_schema
grep -r "from ml.src.preprocessing.graph_schema import" ml/ --include="*.py" -l
grep -r "from .graph_schema import" ml/ --include="*.py" -l
grep -r "import graph_schema" ml/ --include="*.py" -l
```

Known consumers as of v9 schema (verify with search above — this list may be stale):

| File | Uses | Impact if schema changes |
|---|---|---|
| `ml/src/preprocessing/graph_extractor.py` (REAL, 2,056 lines; seam-swap incomplete) | All constants; `_MAX_TYPE_ID` assertion | Re-extraction required; `_MAX_TYPE_ID` must be updated if `max(NODE_TYPES.values())` changes. Note: old path `ml/src/data_extraction/graph_extractor.py` no longer exists. |
| `ml/src/inference/preprocess.py` | All constants | Inference preprocessing must match training; misuse = silent accuracy regression |
| `ml/src/models/gnn_encoder.py` | `NODE_FEATURE_DIM`, `NUM_NODE_TYPES`, `NUM_EDGE_TYPES` | Model constructed with these dims at `__init__`; checkpoint incompatible after change |
| `ml/src/models/sentinel_model.py` | `_MAX_TYPE_ID` assertion, `STRUCTURAL_PREFIX_TYPES` | `_MAX_TYPE_ID` assert fires at import if `max(NODE_TYPES.values())` changes |
| **`data_module/sentinel_data/cli.py` `validate` subcommand** (replaces legacy `ml/scripts/validate_graph_dataset.py`) | `NODE_FEATURE_DIM`, `FEATURE_NAMES`, `EDGE_TYPES` | Validation script will fail or pass incorrectly if not updated |
| `ml/scripts/interpretability/` (all scripts; directory verified exists) | `FEATURE_NAMES` for axis labels | Interpretability plots will have wrong feature names |
| `ml/src/training/training_logger.py` | `NODE_FEATURE_DIM` (imported for `check_inputs`) | WARN alert fires if `graphs.x.shape[-1] != NODE_FEATURE_DIM` |
| `ml/scripts/smoke/` smoke tests | `NODE_FEATURE_DIM`, edge type IDs | Smoke tests may pass with wrong dims if not updated |

---

## J.3 — Migration Procedure

Perform these steps in order. Each step is a gate for the next.

### J.3.1 — Write the ADR First

Every schema change requires an ADR entry before any code changes.
See `docs/ml/adr/` for the existing ADR format. Record:

- What is changing and which constant(s) are affected
- The rationale (what bug or limitation does this fix)
- The migration path (re-extract? re-tokenize? patch in-place?)
- The rollback plan if the change introduces regressions
- Which prior schema version this supersedes

The v8→v9 `in_unchecked_block` re-introduction is the canonical example:
the feature was dropped in v7 as BUG-L2 (87.9% of BCCC is pre-0.8 Solidity,
so the feature appeared dead), then re-introduced in v9 Fix #4 because
IntegerUO detection was unlearnable without it. The ADR for that change is
the model for what belongs in a schema ADR.

### J.3.2 — Update `graph_schema.py`

1. Make the constant change (append only for `NODE_TYPES`/`EDGE_TYPES`)
2. Update `NODE_FEATURE_DIM` if the feature vector length changed
3. Update `NUM_NODE_TYPES` or `NUM_EDGE_TYPES` to match
4. Update `FEATURE_NAMES` to match the new feature layout (index comments must
   reflect actual position)
5. Bump `FEATURE_SCHEMA_VERSION` to the next version string (e.g. `"v9"` → `"v10"`)
6. Update the `SCHEMA HISTORY` docstring with a new version entry following the
   existing format
7. Run the module-level assertions at import to confirm no mismatch:

```bash
PYTHONPATH=. python -c "import ml.src.preprocessing.graph_schema; print('Schema assertions passed')"
```

The three assertions in `graph_schema.py` guard:
- `len(FEATURE_NAMES) == NODE_FEATURE_DIM`
- `len(EDGE_TYPES) == NUM_EDGE_TYPES`
- `len(NODE_TYPES) == <expected count>` and `max(NODE_TYPES.values()) == <expected max>`

All four must pass before proceeding.

### J.3.3 — Update Downstream Consumers

For each file identified in J.2:
1. Update any hardcoded feature indices or node/edge type IDs
2. Update `_MAX_TYPE_ID` in `graph_extractor.py` and `sentinel_model.py` if
   `max(NODE_TYPES.values())` changed (v9 value is `13`; read from source, do
   not assume)
3. Update smoke tests that assert on `NODE_FEATURE_DIM` or edge type IDs
4. Update `validate_graph_dataset.py` expected-dim check if it hardcodes the dim

### J.3.4 — Run Smoke Suite

Run `D.1` from `D_smoke_preflight.md` with the updated code **before** triggering
the full re-extraction. Smoke catches shape mismatches cheaply on a single
contract. A passing smoke suite confirms the updated code is self-consistent.

### J.3.5 — Re-extract All Graphs

Required for all schema changes except runtime-only edge type additions (see J.1
table). Read `data_module/sentinel_data/cli.py` docstring (the `represent` and
`validate` subcommands) before running — the operation is destructive (overwrites
existing `.pt` files in the v3 export directory).

```bash
PYTHONPATH=. python -m sentinel_data.cli represent \
    --graphs-dir data_module/data/exports/sentinel-v3-smartbugs-2026-06-13 \
    --contracts-dir data_module/data/contracts
# (Or use the v4 export directory once Run 13 produces it)
```

**Note**: The legacy `ml/scripts/reextract_graphs.py` no longer exists; it
was replaced by the `data_module` CLI orchestrator.

After re-extraction:
- Run `python -m sentinel_data.cli validate` to confirm all graphs have the new
  `NODE_FEATURE_DIM` and the correct edge shape `[E]` (not `[E, 1]`)
- Confirm the count of extracted graphs matches the prior count ± 0.1%
  (a large drop indicates a Slither compatibility issue with the new schema)

### J.3.6 — Re-tokenize (if required)

Only required if tokenizer logic changed (see J.1 table). Read
`data_module/sentinel_data/cli.py` docstring (the `represent` subcommand) before
running — it is also destructive. Reference `E_preprocessing_consistency.md` E.5
for exact trigger conditions.

**Note**: The legacy `ml/scripts/retokenize_windowed.py` no longer exists; it
was replaced by the `data_module` CLI orchestrator.

### J.3.7 — Schema-Dim Gate Test

Before launching a training run, run `vram_gate_test.py` with the updated
configuration to confirm the GNN encoder initialises with the new dims and
fits in VRAM:

```bash
PYTHONPATH=. python ml/scripts/vram_gate_test.py
```

This catches `GNNEncoder(in_channels=<wrong_dim>)` errors before a full run.

### J.3.8 — Retrain

Follow `F_new_run_checklist.md` fully for the retrain. Record the new schema
version in the run name and MLflow run tags. Do not compare this run's metrics
directly against prior runs — schema changes make metrics non-comparable
(different feature encoding = different effective task).

---

## J.4 — Slither Version Change Policy

A Slither version upgrade is treated as a potential schema change trigger.
Before upgrading:

1. Check if the new version changes any IR node types referenced in
   `graph_extractor.py` (e.g. `NodeType.STARTUNCHECKED`, `HighLevelCall`,
   `LowLevelCall`, `Transfer`, `Send`, `EventCall`)
2. Check if `node.scope.is_checked` behaviour changed (used for `in_unchecked_block` [11])
3. Run `validate_graph_dataset.py` on 100 contracts with the old and new
   Slither version and compare edge counts and feature value distributions

`graph_schema.py` has a hard import-time assertion that Slither >= 0.9.3
(for `NodeType.STARTUNCHECKED` support). If upgrading past a version that
changes this API, update both the assertion and the affected extraction code.

The minimum supported version constraint is in `graph_schema.py`:
```python
if _version < (0, 9, 3):
    raise RuntimeError(...)
```
Read this check before upgrading to confirm the new version is covered.

---

## J.5 — What Is NOT a Schema Change

To prevent unnecessary re-extractions, confirm these are NOT schema changes:

- Changing `gnn_hidden_dim`, `gnn_layers`, `gnn_heads` in `TrainConfig` —
  these are model hyperparameters, not schema constants
- Changing `gnn_prefix_k` or `STRUCTURAL_PREFIX_TYPES` — prefix injection
  is computed at training time from existing `.pt` files
- Adding a new `EDGE_TYPES` entry that is runtime-only (never written to disk,
  e.g. `REVERSE_CONTAINS`=7) — requires only GNNEncoder update, not re-extraction,
  but `NUM_EDGE_TYPES` must still be incremented
- Changing Phase 1/2/3 edge type subsets in `GNNEncoder` (`gnn_phase2_edge_types`
  config) — routing change, not schema change; existing `.pt` files are valid

---

## J.6 — Completion Attestation

After completing a schema migration, append to the relevant ADR document:

```
## Procedure Attestation — J_schema_migration — <ISO date>
Schema version: <old> → <new>
Change: <what changed>
ADR written:                             YES/NO (path)
Steps completed:
  J.3.1 ADR written first:               YES/NO
  J.3.2 graph_schema.py updated:         YES/NO
    FEATURE_SCHEMA_VERSION bumped:        YES/NO
    Module assertions passed:             YES/NO
  J.3.3 downstream consumers updated:    YES/NO
    _MAX_TYPE_ID sites updated:           YES/NO/N/A
    smoke tests updated:                  YES/NO
  J.3.4 smoke suite passed:              YES/NO
  J.3.5 graphs re-extracted:             YES/NO/N/A (reason if N/A)
    validate_graph_dataset passed:        YES/NO
    graph count delta:                    <old> → <new> (±N)
  J.3.6 tokens re-extracted:             YES/NO/N/A
  J.3.7 vram gate passed:                YES/NO
  J.3.8 retrain launched:                YES/NO
Steps skipped:     [any skipped + explicit reason]
Rollback plan:     [documented in ADR]
Written to:        [path of this attestation]
```
