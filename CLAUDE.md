# CLAUDE.md — Branch Context

## Branch Goal

This branch (`claude/festive-lamport-qJmtj`) is a structured self-study resource for mastering the SENTINEL ML codebase for interviews across four tracks: ML, AI, MLOps, and Blockchain/Solidity.

All teaching materials live in `ml/ml_learning_with_claude/`.

## Spec Files

- **`ml/ml_learning_with_claude/ROADMAP.md`** — module-by-module learning plan, chunk index, what's done vs pending, interview coverage map. Check here first to orient.
- **`ml/ml_learning_with_claude/learning_preferences.md`** — 14 active teaching preferences (P1–P14). Every new teaching chunk must comply with all of them. Read before writing any new chunk.

## Current State

Modules done: Preprocessing (6 chunks), DataExtraction (1), Datasets (1), Models (5 — all of `gnn_encoder.py`, `transformer_encoder.py`, `fusion_layer.py`, `sentinel_model.py`).

Next up: **Training module** (`ml/src/training/`) — `trainer.py` (1,633 lines), `losses.py`, `focalloss.py`.

## Key Rule

P1–P14 from `learning_preferences.md` apply to all new chunks from Training onwards. Existing chunks pre-date the spec and are not retroactively updated.
