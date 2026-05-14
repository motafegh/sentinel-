# SENTINEL ML Module — Documentation Index

This directory contains the complete documentation for the SENTINEL ML module: a dual-path GNN + CodeBERT architecture for multi-label smart contract vulnerability detection.

The documentation is split into two groups that complement each other:

---

## Group 1 — Technical Reference

These documents answer **"what exactly does it do?"** They are precise, complete, and authoritative. Use them when you need to verify a constant, understand an interface, or write code that interacts with the system.

| Document | Purpose |
|---|---|
| [ML-T1-ARCHITECTURE.md](ML-T1-ARCHITECTURE.md) | Every module, layer, constant, and parameter in the model. The authoritative source for locked constants (NODE_FEATURE_DIM, NUM_EDGE_TYPES, fusion_output_dim, etc.). Start here if you need to know a shape, a dim, or a constraint. |
| [ML-T2-TRAINING-PIPELINE.md](ML-T2-TRAINING-PIPELINE.md) | Every TrainConfig field, optimizer param groups, scheduler setup, loss functions, monitoring metrics, checkpoint format. Start here if you need to change how training works. |
| [ML-T3-DATA-PIPELINE.md](ML-T3-DATA-PIPELINE.md) | Dataset layout on disk, hash identity system, graph/token file formats, deduplication history, CEI augmentation, splits, RAM cache. Start here if you need to understand or modify the data pipeline. |

---

## Group 2 — Educational Learning

These documents answer **"why does it work this way?"** They are narrative, contextual, and instructional. Use them when you need to understand the motivation behind a decision, debug a training run, or onboard to the project.

| Document | Purpose |
|---|---|
| [ML-E1-DESIGN-RATIONALE.md](ML-E1-DESIGN-RATIONALE.md) | The story behind every major architectural decision: why GNN+CodeBERT, why three-phase GNN, why JK connections (the gradient collapse story), why REVERSE_CONTAINS, why per-phase LayerNorm, why function-level pooling (the CFG_RETURN flood story), why separate LR groups, why behavioral tests are the real judge. Start here to understand the system before modifying it. |
| [ML-E2-TRAINING-GUIDE.md](ML-E2-TRAINING-GUIDE.md) | Step-by-step practitioner's guide: environment setup, pre-training checklist, smoke run → full run workflow, how to read training logs, success and abort criteria, threshold tuning, behavioral testing, resume patterns, common failure modes and fixes. Start here when you need to actually run training. |

---

## How the Documents Relate

```
Want to RUN training?
  → E2-TRAINING-GUIDE      (step-by-step commands, gates, what to watch)
    └─ T2-TRAINING-PIPELINE (full TrainConfig reference when you need exact field names)

Want to MODIFY the model?
  → E1-DESIGN-RATIONALE    (understand why it's designed this way before changing it)
    └─ T1-ARCHITECTURE      (exact specs: shapes, dims, locked constants)

Want to understand the DATA?
  → T3-DATA-PIPELINE       (complete data reference: formats, counts, splits)
    └─ E1-DESIGN-RATIONALE §Deduplication (why the dataset was halved — critical context)

Debugging a training failure?
  → E2-TRAINING-GUIDE §Failure Modes
    └─ E1-DESIGN-RATIONALE §JK Connections (if GNN share collapses)
    └─ T2-TRAINING-PIPELINE §Monitoring (exact metric names in MLflow)
```

---

## Quick Facts (as of v5.2)

| Item | Value |
|---|---|
| Architecture | Three-eye: GNNEncoder (3-phase GAT) + CodeBERT/LoRA + CrossAttentionFusion |
| Task | Multi-label, 10 vulnerability classes |
| Dataset | 44,470 contracts (deduplicated from 68K) |
| Splits | train=31,142 / val=6,661 / test=6,667 |
| NODE_FEATURE_DIM | 12 (locked) |
| NUM_EDGE_TYPES | 8 (REVERSE_CONTAINS=7 is runtime-only) |
| NUM_CLASSES | 10 (locked, append-only) |
| fusion_output_dim | 128 (locked — ZKML depends on it) |
| GNN LR | base × 2.5 (gradient collapse countermeasure) |
| LoRA LR | base × 0.5 (catastrophic forgetting prevention) |
| Best checkpoint | v5.0: `ml/checkpoints/v5-full-60ep_best.pt` (tuned F1=0.5828, behavioral FAILED) |
| Active fallback | v4: `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt` (tuned F1=0.5422) |
| v5.2 status | Code complete; smoke run pending |

---

## Current State (2026-05-14)

All Phase 0/1/2 code changes for v5.2 are complete and tested (11/11 GNN encoder tests pass). The next step is the **Phase 4 smoke run**:

```bash
source ml/.venv/bin/activate
export TRANSFORMERS_OFFLINE=1
PYTHONPATH=. python ml/scripts/train.py \
    --run-name v5.2-smoke \
    --experiment-name sentinel-v5.2 \
    --epochs 2 \
    --smoke-subsample-fraction 0.1 \
    --gradient-accumulation-steps 4
```

See [ML-E2-TRAINING-GUIDE.md](ML-E2-TRAINING-GUIDE.md) for the full workflow and success gates.
