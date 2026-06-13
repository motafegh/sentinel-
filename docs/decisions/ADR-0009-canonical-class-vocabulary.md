# ADR-0009: Canonical 10-Class Vocabulary — Labeling Order as the Single Source of Truth

**Date:** 2026-06-12
**Status:** ACCEPTED
**Author:** SENTINEL v2 audit (Phase D)
**Supersedes:** the implicit "two-taxonomy" divergence documented in `data_module/README.md` (which is now stale)

---

## Context

SENTINEL classifies Solidity contracts across 10 vulnerability classes. The class
**order** is critical: model outputs, label CSVs, parquet columns, and the
training loss must all align by index, otherwise the model trains on shuffled
labels (Run 11 silent corruption risk).

Pre-Phase D, the codebase had TWO class orderings in circulation:

| Source | Order |
|---|---|
| `ml/src/preprocessing/graph_schema.py` (legacy canonical, pre-7B) | Representation order: `Reentrancy=0, CallToUnknown=1, Timestamp=2, ExternalBug=3, GasException=4, DoS=5, IntegerUO=6, UnusedReturn=7, MishandledException=8, NonVulnerable=9` (9 vulns + NonVulnerable, **no TransactionOrderDependence**) |
| `sentinel_data/labeling/schema/taxonomy.yaml` (labeling order) | `CallToUnknown=0, DoS=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9` (9 vulns + UnusedReturn, **no NonVulnerable slot**) |
| `ml/src/training/trainer.py:105` (defines own copy) | LABELING order — same as `taxonomy.yaml` |
| v9 best checkpoint `class_names` field | LABELING order |
| v2 export `labels.parquet` columns `class_0..class_9` | LABELING order |
| `sentinel_data/labeling/schema.class_names()` | LABELING order (by construction) |

The seam swap (Stage 7B) moved the canonical to
`sentinel_data/representation/graph_schema.py` and declared it the "single
source of truth." But the new canonical was a copy of the **representation
order** (not the labeling order). The trainer and checkpoint were already in
labeling order, so the v2 export + trainer + checkpoint were CONSISTENT. The
canonical was the OUTLIER.

The README at `data_module/README.md:218-223` flagged this as a known issue.

---

## Decision

**The canonical 10-class order is the LABELING order.** This is the order used
in production by every component that ships or trains:

- `ml/src/training/trainer.py:105-116`
- `ml/src/preprocessing/graph_schema.py` (shim re-exports the canonical)
- The v9 best checkpoint `class_names` field
- The v2 export `labels.parquet` columns
- `sentinel_data.labeling.schema.class_names()`
- (Post-fix) `sentinel_data/representation/graph_schema.py:CLASS_NAMES`

Concretely: `CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3,
IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7,
TransactionOrderDependence=8, UnusedReturn=9`.

The "representation order" (with NonVulnerable at index 9) is **historical** and
no longer used in production. It is not a valid alternative ordering.

---

## Why this is the right decision

1. **Every active production component uses the labeling order.** The trainer
   was last edited at commit `e2ad84e` (Run 7+). The v9 best checkpoint
   (GCB-P1-Run9-v11-20260606_best.pt) has `class_names` in labeling order. The
   v2 export was created with labeling order. The format_schema/v1.yaml
   documents labeling order.

2. **The seam swap declared the data module's canonical as the source of
   truth.** If we leave it in representation order, any future code that
   imports `CLASS_NAMES` from `sentinel_data.representation.graph_schema`
   (the canonical) gets the WRONG order — silently breaking training.

3. **The labeling order has 10 vulnerability classes** including
   TransactionOrderDependence (which is a real DASP-10 class). The
   representation order omits TransactionOrderDependence and has
   NonVulnerable at index 9 instead. Since NonVulnerable is the
   "all-zero" negative case, it doesn't need a class index; the
   absence of any positive label is the negative.

4. **The change is SAFE.** The canonical is not imported by any production
   training code. The trainer, predictor, checkpoint, and export all use
   their own copies of the labeling order. Updating the canonical to match
   the labeling order has no runtime effect — it only fixes the docstring.

5. **The change is ENFORCED.** The
   `tests/test_representation/test_thin_adapter.py::test_class_names_and_num_classes_are_present`
   test was updated to assert the labeling order. Future refactors that
   re-introduce the representation order will fail the test.

---

## What changed in this ADR

1. **`sentinel_data/representation/graph_schema.py:190-201`** — `CLASS_NAMES`
   updated to labeling order. `NonVulnerable` removed; `TransactionOrderDependence`
   inserted at position 8. The block has a comment explaining the canonical
   status and pointing to this ADR.

2. **`tests/test_representation/test_thin_adapter.py:104-128`** — the
   class-name ordering test was updated to assert the labeling order. The
   old assertions (`CLASS_NAMES[0] == "Reentrancy"`,
   `CLASS_NAMES[9] == "NonVulnerable"`) were replaced with the 6
   labeling-order assertions.

3. **`data_module/README.md:218-223`** — the "TWO definitions in the
   codebase" warning was REPLACED with a "single source of truth" note
   that documents the labeling order, points to this ADR, and lists the
   6 places that all use the same order.

4. **`ml/src/datasets/sentinel_dataset.py` and the v2 export** — unchanged.
   They were already in labeling order and continue to work.

5. **`ml/src/training/trainer.py:105-116`** — unchanged. Was already in
   labeling order.

6. **`ml/src/preprocessing/graph_schema.py`** — unchanged. It's a shim that
   re-exports from the canonical; updating the canonical automatically
   propagates to all consumers (`gnn_encoder.py`, `sentinel_model.py`,
   `predictor.py`, etc.). None of those consumers import `CLASS_NAMES`
   today, but the assertion is safe.

---

## Validation

- `pytest tests/` → **575 passed, 47 skipped, 0 failed** (was 535/51/0 before
  this fix; +40 net tests now visible because some skips were
  conditionals on the canonical being loadable through the import chain).
  Actually net: same 535→575 = +40 because the F1 fix's
  `test_real_dive_csv_sample` and the new v2 export made
  `test_sentinel_dataset.py` tests run instead of skip.

- `SentinelDatasetExport(...).verify_artifact_hash()` → True
- `SentinelDataset("train", ...)` loads cleanly with the v9 best checkpoint
- The v9 best checkpoint's `class_names` matches the new canonical
- The v2 export's `labels.parquet` columns match the new canonical
- The trainer's `CLASS_NAMES` matches the new canonical (it was already
  correct)

---

## Risk: what could go wrong

1. **Someone re-introduces the representation order.** Mitigated by the
   regression test in `test_thin_adapter.py`. The test must be kept in
   sync if the order ever legitimately changes (which requires
   re-training Run 11+ from scratch and bumping `FEATURE_SCHEMA_VERSION`).

2. **An old checkpoint from pre-Run-7 is loaded.** Such a checkpoint
   would have representation-order `class_names`. The predictor's
   "Strict checkpoint metadata validation" (Fix from 2026-05-04) would
   log a warning. The model would also produce wrong results because its
   classifier head would have the wrong number of outputs (it expects 10
   vulnerability classes; the representation order had NonVulnerable
   instead of TransactionOrderDependence). This is a known limitation;
   pre-Run-7 checkpoints are explicitly out of support.

3. **A new bug is introduced that shuffles labels between the trainer
   and the export.** The schema-dim gate in
   `tests/test_representation/test_byte_identical_regression.py` (Stage 2)
   and the new `test_thin_adapter.py` assertions catch this at test
   time. In production, the 3 gates in `SentinelDataset.__init__`
   (format schema version, graph schema version, artifact hash) catch
   format mismatches but NOT class-order mismatches. A future
   improvement: add a 4th gate that compares the export's
   `label_class_columns` against the trainer's `CLASS_NAMES` and raises
   on mismatch. Tracked as a follow-up; not a Run 11 blocker.

---

## Follow-up

- [ ] Add 4th gate to `SentinelDataset.__init__` that asserts
      `manifest.label_class_columns == CLASS_NAMES` (the trainer's). This
      catches class-order mismatches at load time, not test time.
- [ ] When the v2.1 corpus adds DeFiHackLabs / Web3Bugs / SmartBugs
      Curated, re-verify the canonical order against the new labeled data
      distribution.
- [ ] If the 10-class taxonomy ever changes (e.g., adding a new class),
      bump `FEATURE_SCHEMA_VERSION` to "v10" AND re-train Run N+1 from
      scratch. Do NOT silently extend the 10-class list.

---

## References

- `data_module/audit/v2_full_audit/01_phase_a_foundation_recon.md` — FINDING-A:11 (original two-taxonomy flag)
- `data_module/audit/v2_full_audit/04_phase_c2_stage_7_export_audit.md` — FINDING-C2:1 (latent footgun discovered)
- `data_module/sentinel_data/representation/graph_schema.py:190-213` (updated by this ADR)
- `data_module/tests/test_representation/test_thin_adapter.py:104-128` (updated by this ADR)
- `ml/src/training/trainer.py:105-116` (already in labeling order, unchanged)
- `data_module/docs/legacy/bccc_deep_dive/Phase5_LabelVerification_2026-06-08/` (the v1.3/v1.4 labels that introduced the labeling order)
- MEMORY.md §"Sentinel v2 Data Module Build" (Stage 3 ships reference to the labeling order)
