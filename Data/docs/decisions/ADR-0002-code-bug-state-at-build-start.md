# ADR-0002 — Code bug state at build start

**Date:** 2026-06-09
**Status:** Accepted
**Deciders:** Ali Motafegh

---

## Context

The pre-Run-8 audit (`docs/pre-run-fixes/validated_audition.md`) identified 36 code issues (A1–A38) across 9 source files. By the time Stage 0 starts (2026-06-09), 8 of those bugs are fixed and must be preserved through the port. 3 bugs are still open and are scheduled for Stage 7.

This ADR catalogs the state so that Stage 2 (port) and Stage 7 (seam swap) know exactly what the regression test must guard and what it must fix.

---

## 8 bugs already fixed — do NOT re-fix

The Stage 2 byte-identical regression test guards all of these. Re-fixing them would be a regression.

| Bug | File:line | Original | Fix | Regression test |
|---|---|---|---|---|
| **A9** `now` keyword miss | `ml/src/preprocessing/graph_extractor.py:587-605` | `_compute_uses_block_globals` checked `SolidityVariableComposed` isinstance only — missed Solidity 0.4.x `now` keyword | Added `getattr(rv, "name", "") == "now"` check | Stage 2 byte-identical test |
| **A15** def_map by name | `ml/src/preprocessing/graph_extractor.py:1147-1179` | `def_map` keyed by variable name (string) — shadowing in nested scopes produced spurious DEF_USE edges | Changed to two-tier scope key using `id(lval)` | Stage 2 byte-identical test |
| **A20** label=0 hardcode | `ml/src/data_extraction/ast_extractor.py:290,342,395` | `batch extraction used `partial(self.contract_to_pyg, ..., label=0)` — all contracts got `graph.y = tensor([0])` | `label_map` from CSV; `_labeled_pool_worker`; `--label-csv` CLI arg | Stage 1 A20 regression test |
| **A34** prefix sort dim | `ml/src/models/sentinel_model.py:356,367` | `select_prefix_nodes` secondary sort used post-GAT embedding dim 10, not raw `external_call_count` | Changed to sort by raw feat[3] | Stage 2 byte-identical test |
| **A38** NaN before backward | `ml/src/training/trainer.py` | `loss.backward()` ran before `torch.isfinite(loss)` check — NaN gradients could permanently corrupt Adam m2 buffers | `isfinite` check moved before `backward()`; post-clip guard; gate at >0.5% NaN rate | (training concern; no data-path regression test needed) |
| **Resume overwrite** | `ml/src/training/trainer.py:383,1184,1206,1212` | `resume_model_only=True` default caused ep14 best checkpoint to be silently overwritten by ep1 weights | `resume_model_only=False` as default; timestamped `--run-name` | Stage 8 launch checklist |
| **return_ignored** | `ml/src/preprocessing/graph_extractor.py` | `_compute_return_ignored` checked `op.lvalue is None` — Slither always creates TupleVariable/TemporaryVariable; feat[7]=0.0 universally | Changed to check if `id(lvalue)` appears in subsequent IR ops | Stage 2 byte-identical test |
| **EMITS edge** | `ml/src/preprocessing/graph_extractor.py` (Interp-6) | Emits edge logic was incorrect | Fix was applied; see Interp-6 notes | Stage 2 byte-identical test |

---

## 3 bugs still open — fix in Stage 7

| Bug | File:line | Root cause | Stage 7 action |
|---|---|---|---|
| **CALL_ENTRY cross-function for external calls** | `ml/src/preprocessing/graph_extractor.py:1001` | `_add_icfg_edges()` iterates `node.internal_calls` only — external calls (HighLevelCall/LowLevelCall) get no cross-function CALL_ENTRY edge | Stage 2 port preserves partial fix; full cross-function fix is post-Run-11 (v2.1) |
| **Predictor tier threshold** | `ml/src/inference/predictor.py:150,168,752` | `TIER_CONFIRMED_THRESHOLD = 0.55` hardcoded; tuned per-class thresholds from `{ckpt}_thresholds.json` loaded into `self.thresholds` but never consulted in tier logic | Stage 7 seam swap fixes — use tuned per-class thresholds everywhere |
| **99% DoS↔Reentrancy co-occurrence** | Source: BCCC folder-based labeling | 99% of DoS-labeled contracts are also Reentrancy-labeled, because BCCC stores the same `.sol` under multiple class dirs | Stage 3 merger flags and quantifies; Stage 6 co-occurrence matrix reports; BCCC is deferred so v2 corpus won't have this |

---

## How downstream stages use this ADR

- **Stage 2 (representation port):** The byte-identical regression test scope covers all 9 source files. The test verifies that A9, A15, A20, A34, A38, return_ignored, and EMITS fixes are preserved. If any test fails, the port is incomplete.
- **Stage 7 (seam swap + export):** Must fix the 3 open bugs before the seam swap gate. The dual-path seam swap test verifies all 8 fixed bugs are preserved and all 3 open bugs are patched.
- **Stage 8 (Run 11 launch):** Confirms the resume-overwrite fix is active (`resume_model_only=False` default); confirms the watcher `F1 > 0.1` floor is in place.
