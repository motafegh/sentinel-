# Pre-Run-8 Deep Investigation Plan

**Date:** 2026-06-04  
**Time started:** current session  
**Author:** Investigation session (Claude Code)

---

## Purpose

The three reference documents (`GCB-P1-Run7-analysis-2026-06-04.md`, `SENTINEL-Understanding-Run7.md`, `EXPERIMENT_INDEX.md`) are starting points, NOT ground truth. This investigation reads source code directly to:

1. Verify every claim made in those documents against actual code
2. Discover problems/bugs those documents missed
3. Identify improvements beyond the planned Run 8 changes
4. Produce a checklist that ensures the human fully understands each issue

---

## Investigation Areas

| ID | Area | Source File(s) | Status |
|----|------|---------------|--------|
| **INV-1** | `--drop-complexity-feature` implementation readiness | `gnn_encoder.py`, `train.py` | ⚠️ NOT IMPLEMENTED — F1 |
| **INV-2** | GNN forward pass — phase gating, edge filtering, skip connections | `gnn_encoder.py` | ✅ NF-6 fix confirmed; edge masks correct |
| **INV-3** | JK attention and entropy reg — correctness | `gnn_encoder.py` | ✅ H_max=log(3) correct; entropy formula correct |
| **INV-4** | Four-eye classifier wiring — correct gradient flow | `sentinel_model.py` | ✅ CFG eye direct path correct |
| **INV-5** | CrossAttentionFusion — fusion_max_nodes truncation (BUG-C4) | `sentinel_model.py` | ⚠️ 227 graphs (0.55%) >1024; 0 >2048; fix=flag only |
| **INV-6** | Trainer — loss aggregation, AMP scaler, optimizer groups | `trainer.py` | ✅ JK reg inside autocast; window normalisation correct; F2/F3 bugs found |
| **INV-7** | Trainer — threshold tuning correctness | `trainer.py` | ✅ Correct; _is_final_epoch heuristic is conservative-only (F11) |
| **INV-8** | Trainer — checkpoint save/restore completeness | `trainer.py` | ✅ Fix#35 RNG states + thresholds saved; config includes all fields |
| **INV-9** | train.py argparse — missing flags, wiring errors | `train.py` | ⚠️ `aux_cei_loss_weight` silently discarded (F3); sampler default mismatch (F2) |
| **INV-10** | StructuredLogger — post-BUG-SL-1 fix correctness sweep | `training_logger.py` | ✅ Fix confirmed at line 304 |
| **INV-11** | gnn_prefix injection — disabled in Run 7, re-enable path | `sentinel_model.py`, `gnn_encoder.py` | ✅ Path works; F6 compile list miss is perf-only |
| **INV-12** | Calibration pipeline — does calibrate_thresholds.py exist? | `ml/calibration/` | ⚠️ No run7 thresholds file; extract from checkpoint["tuned_thresholds"] |
| **INV-13** | BUG-C4 quantification — how many graphs >1024 nodes? | `ml/data/splits/` | ✅ 227 (0.55%) >1024; 0 >2048; max=1735; use --fusion-max-nodes 2048 |

---

## Output Documents

- **This file** — plan and progress tracker
- `UNDERSTANDING-CHECKLIST.md` — running checklist for human understanding
- `FINDINGS.md` — detailed findings per investigation area
- `RUN8-ULTRACODE.md` — updated with new findings

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not started |
| 🔍 | In progress |
| ✅ | Complete — no issues |
| ⚠️ | Complete — issues found |
| 🐛 | Bug confirmed |
| 💡 | Improvement identified |

---

## Progress Log

- [x] INV-1: drop-complexity-feature — NOT IMPLEMENTED (see FINDINGS F1, code changes pending)
- [x] INV-2: GNN forward pass — CLEAN
- [x] INV-3: JK attention + entropy reg — CLEAN
- [x] INV-4: Four-eye classifier wiring — CLEAN
- [x] INV-5: fusion_max_nodes / BUG-C4 — QUANTIFIED (see FINDINGS F5)
- [x] INV-6: Trainer loss + optimizer — 2 bugs found (F2, F3)
- [x] INV-7: Trainer threshold tuning — CLEAN (minor heuristic, F11)
- [x] INV-8: Trainer checkpoint save/restore — CLEAN
- [x] INV-9: train.py argparse — 2 bugs found (F2, F3)
- [x] INV-10: StructuredLogger post-fix sweep — BUG-SL-1 FIX CONFIRMED
- [x] INV-11: gnn_prefix injection path — CLEAN (perf miss F6)
- [x] INV-12: Calibration pipeline — MISSING (see FINDINGS F10)
- [x] INV-13: BUG-C4 graph size count — DONE (227 graphs >1024, 0 >2048, max=1735)

**Investigation complete. Findings in FINDINGS.md. Code changes in progress.**
