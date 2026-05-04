# Changelog Index

One-line summary per dated changelog. See individual files for full details.

| Date | File | Summary |
|------|------|---------|
| 2026-04-29 | [2026-04-29-contracts-restructure-and-proxy-fix.md](2026-04-29-contracts-restructure-and-proxy-fix.md) | Contracts restructured to Foundry layout; UUPS proxy pattern applied to AuditRegistry |
| 2026-04-30 | [2026-04-30-readme-ml-hardening.md](2026-04-30-readme-ml-hardening.md) | README updated; ML model hardening (AMP, TF32, warmup pass, checkpoint validation) |
| 2026-05-01 | [2026-05-01-ml-audit-hardening.md](2026-05-01-ml-audit-hardening.md) | 8 ML audit fixes: pickle security, BF16 cast, warmup 2-node graph, grad clip, LR resume |
| 2026-05-01 | [2026-05-01-ml-next-improvements-plan.md](2026-05-01-ml-next-improvements-plan.md) | Original ML improvement plan (T1-A/B/C/D, T2-A/B/C, T3-A/B) |
| 2026-05-01 | [2026-05-01-shared-preprocessing-architecture.md](2026-05-01-shared-preprocessing-architecture.md) | §5.10 refactor: shared preprocessing package (graph_schema.py, graph_extractor.py); T1-C windowed tokenization in preprocess.py |
| 2026-05-01 | [2026-05-01-sprint-2-4-foundry-ml-rag.md](2026-05-01-sprint-2-4-foundry-ml-rag.md) | Sprint 2–4: Foundry test suite, ML improvements, RAG/agent wiring |
| 2026-05-02 | [2026-05-02-next-movements-plan.md](2026-05-02-next-movements-plan.md) | Strategic plan: Phase 0 pre-retrain, Move 1 T1-C, Move 2 T1-A, Phase 2 observability |
| 2026-05-02 | [2026-05-02-pre-execution-audit-gaps.md](2026-05-02-pre-execution-audit-gaps.md) | Pre-execution audit: 7 gaps corrected in STATUS + ROADMAP before Moves 3–8 proceed |
| 2026-05-02 | [2026-05-02-multi-contract-parsing-gap.md](2026-05-02-multi-contract-parsing-gap.md) | Multi-contract parsing tracked as Move 9; reviewer proposal audited and corrected against source code |
| 2026-05-03 | [2026-05-03-graph-reextraction.md](2026-05-03-graph-reextraction.md) | Full graph dataset re-extraction (68,523 files); edge_attr shape [E,1]→[E] for P0-B; missing deps added to pyproject.toml |
| 2026-05-03 | [2026-05-03-training-pipeline-fix.md](2026-05-03-training-pipeline-fix.md) | create_splits.py stratification fix; full pipeline regeneration (multilabel_index, tokens, splits); retrain launched (multilabel-v2-edge-attr, 40 epochs) |
| 2026-05-04 | [2026-05-04-resume-fixes-and-autoresearch.md](2026-05-04-resume-fixes-and-autoresearch.md) | Fix #9 AttributeError on resume (config.architecture); --no-resume-model-only CLI flag; retrain extended 40→60ep full-resume; autoresearch strategy drafted |
| 2026-05-04 | [2026-05-04-resume-batch-size-fix.md](2026-05-04-resume-batch-size-fix.md) | Fix #11 patience_counter persistence; Fix #12 batch-size guard on full resume; Fix #13 pos_weight warning; --resume-reset-optimizer flag |
| 2026-05-04 | [2026-05-04-post-training-audit-fixes.md](2026-05-04-post-training-audit-fixes.md) | Fix #1–#7 (dual_path_dataset edge_attr, predictor arch args, warmup edge_attr, tune_threshold args + fusion_dim, API thresholds key, arch dim fallback); Fix #9 MLflow focal params |
| 2026-05-04 | [2026-05-04-batch3-resume-fixes.md](2026-05-04-batch3-resume-fixes.md) | Fix #23 patience_counter JSON sidecar; Fix #24 missing optimizer key warning on full resume; Fix #25 explicit total_steps check in scheduler restore |
