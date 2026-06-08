# SENTINEL — Project Changelog

**Scope:** Full project history from initial commit through Phase 5 (Label Verification — 46,977 BCCC labels dropped, v1.3/v1.4 verified dataset produced).
**Last updated:** 2026-06-08

This document is the single authoritative changelog. Session-level detail lives in `docs/changes/` and `docs/ml/`. This file records *what changed, why, and what it produced* — not how to reproduce it.

---

## Topical Reference Table

Quick-jump to any topic. Each row links to the section anchor. For chronological per-day entries (one-line summaries), see [`docs/changes/INDEX.md`](changes/INDEX.md). For the *why* behind current architecture, see [`docs/ml/adr/INDEX.md`](ml/adr/INDEX.md).

### 1. Project Foundation

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [1](#1-project-foundation) | 2026-04-26 → 04-29 | Project Foundation | Dual-path GNN+CodeBERT concept; EZKL/Groth16 ZKML; LangGraph 5-agent topology; Foundry contracts |

### 2. Schema Evolution

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [2](#2-v4-baseline) | pre-2026-05-09 | v4 Baseline | Pre-v5 schema (legacy `NODE_FEATURE_DIM`); 68K rows; 34.9% cross-split leakage |
| [7](#7-v6--graph-feature-schema-patch) | 2026-05-17 | v6 — Graph Feature Schema Patch | BUG-1/2/3 in-place patch on 44,470 graphs; `VISIBILITY_MAP` int→float; `FEATURE_SCHEMA_VERSION` v5→v6 |
| [8](#8-v7--full-architecture-overhaul) | 2026-05-18 – 05-19 | v7 — Full Architecture Overhaul | All 27 bugs fixed; 41,522 graphs re-extracted; 41,577 pairs cached (2.28 GB); 12 config/default misalignments |
| [9](#9-v8--cross-function-graph-extension) | 2026-05-19 – 05-21 | v8 — Cross-Function Graph Extension | ICFG-Lite (`CALL_ENTRY`+`RETURN_TO`) + `DEF_USE` edges; schema v8 (`NUM_EDGE_TYPES=11`); 41,576 graphs |
| [39](#39-v9-schema-upgrades) | 2026-06-06 | v9 Schema Upgrades | Adds `CFG_NODE_ARITH=13`, `EXTERNAL_CALL=11`, `in_unchecked_block` feat[11], `uses_block_globals` extension; v8→v9 |

### 3. Architecture, Loss & Model Design

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [3](#3-v50--three-eye-architecture) | 2026-05-11 – 05-12 | v5.0 — Three-Eye Architecture | GNN+CFG+TF three-eye design with aux loss (F1=0.27 plateau; complexity proxy) |
| [5](#5-v52--jk--lora--three-phase-gnn) | 2026-05-14 – 05-16 | v5.2 — JK + LoRA + Three-Phase GNN | JK attention, per-phase LayerNorm, `REVERSE_CONTAINS` type-7, separate LR groups, NaN counter |
| [6](#6-v53--asl-loss-experiment) | 2026-05-16, killed | v5.3 — ASL Loss Experiment | `pos_weight_min_samples=3000` over-corrected; killed ep47 F1=0.2559; decision: fix schema first |
| [13](#13-phase-36--graphcodebert--gnn-prefix-injection) | 2026-05-23 – 05-24 | Phase 3.6 — GraphCodeBERT + GNN Prefix Injection | 124M GraphCodeBERT + LoRA r=16 α=32 on Q+V; `gnn_prefix_k=0` in Run 7 (disabled) |
| [16](#16-three-tier-ml-output) | 2026-05-27 | Three-Tier ML Output | CONFIRMED/SUSPICIOUS/NOTEWORTHY schema; predictor + API integration |
| [28](#28-run-5-training-log-specification) | 2026-06-02 | Run 5 Training Log Specification | 7-section log spec: 3A core, 3B AUC/calibration, 4 system, 5 interventions |
| [35](#35-run-7-architecture--issue-14-fixes) | 2026-06-03 | Run 7 Architecture + ISSUE-1–4 Fixes | 4-eye (BUG-R7-1/2, IMP-R7-1/2/3), type emb, Ph2 heads=4, aux warmup 8ep, 0.005 entropy reg |
| [ADR-0002](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0002 — Multi-label formulation | 10-class sigmoid multi-hot; class_9 reserved (SENTINEL phase 2) |
| [ADR-0003](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0003 — Dual-path Four-Eye architecture | GNN + LoRA-CodeBERT cross-attend; 4 eyes (GNN/TF/Fused/CFG) summed for final logits |
| [ADR-0004](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0004 — Three-phase GAT routing | 8-layer GAT split Ph1/Ph2/Ph3; Ph2 sub-routing for `EXTERNAL_CALL` |
| [ADR-0006](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0006 — Loss formulation | ASL γ⁻=2 γ⁺=1 per-eye + aux BCE pathway (0→0.30 over 8ep) + 0.005 JK entropy |

### 4. Training Runs

| § | Date | Run | Outcome |
|---|------|-----|---------|
| [2](#2-v4-baseline) | pre-2026-05-09 | v4 Baseline | tuned F1=0.5422 (leakage-inflated, not reliable) |
| [5](#5-v52--jk--lora--three-phase-gnn) | 2026-05-14 – 05-16 | v5.2-jk-20260515c-r2 | Killed at 0-byte log; restarted; aux-loss issue identified |
| [6](#6-v53--asl-loss-experiment) | 2026-05-16 | v5.3 | Killed ep47 F1=0.2559 (pos_weight over-correction) |
| [10](#10-v8-ab--joint-icfgdef_use-ablation) | 2026-05-20 | v8-AB | ep26 F1 plateau 0.236–0.259; killed at ep37 (patience 8/30); H1/H5 confirmed |
| [11](#11-plan-3a--icfg-only-ablation) | 2026-05-21 – 05-23 | PLAN-3A — ICFG-Only | F1=0.2877 (better than v8-AB) but still below Run 4 |
| [14](#14-imp--architectural-fixes--p1-train-run-2) | 2026-05-24 | P1-TRAIN Run 2 | 8-layer GNN, IMP-G1/G2/G3/M1/M2T2/M3/D1; 134/134 tests pass; launched PID 80610 |
| [15](#15-p1-train-runs-3-and-4) | 2026-05-25 – 05-26 | P1-TRAIN Runs 3+4 | Run 3 ep16 F1=0.17 (double-amp bug); **Run 4 ep32 F1=0.3362** (best, capacity ceiling ep44) |
| [36](#36-fix-35--safe-resume-rng-state--full-optimizer-restore) | 2026-06-04 | Run 7 (`GCB-P1-Run7-v10-20260603`) | ep39 F1=0.3074 fixed / **0.3423 tuned** (target to beat) |
| (run record) | 2026-06-04 | Run 8 (`GCB-P1-Run8-v10-20260605`) | Killed ep29 KeyboardInterrupt; best ep27 val F1=0.2814; test tuned F1=**0.2307** (regression); see [§37](#37-pre-run-9-audit-findings) |
| [40](#40-run-9-launch--watcher) | 2026-06-06 | **Run 9** (`GCB-P1-Run9-v11-20260606`) | In flight, ep14 at last check; best F1=0.2476 at ep12; top3 IntegerUO/GasException/MishandledException |
| [42](#42-run-9-v11--crash--resume--lambda-typo) | 2026-06-06 | **Run 9 v11** crash + resume + lambda typo | ep16 crash (VS Code close 15:49); 1st resume had JK lambda typo (0.0075 vs 0.005); ep14 best (F1=0.2586) lost as .pt; 2nd resume (lambda 0.005) in flight ~17:23 |

### 5. Data Pipeline & Quality

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [4](#4-v51--dataset-deduplication) | 2026-05-12 | v5.1 — Dataset Deduplication | 68K→44K rows; 34.9% cross-split leakage eliminated; clean splits in `splits/deduped/` |
| [12](#12-phase-35--data-quality-fixes) | 2026-05-23 | Phase 3.5 — Data Quality Fixes | DQ-1 stricter label cleaning, DQ-2 DoS gradient re-enabled, DQ-3 complexity diagnostic, H5 refutation |
| [23 (line 1195)](#23-pre-run-5-implementation--label-cleaning--cei-aux-loss--calibration) | 2026-05-31 | Pre-Run-5 Implementation | Label cleaning scripts, CEI aux loss, temperature scaling |
| [34](#34-v10-re-extraction-launch--script-defaults-alignment) | 2026-06-02 | v10 Re-Extraction Launch | 41,576 graphs re-extracted; cache 2.5 GB; 0 overlap splits 29,103/6,236/6,237 |
| [38](#38-pre-run-9-fixes) | 2026-06-06 | Pre-Run 9 Fixes | #1 relabel-timestamp, #2 block-globals, #3 external CALL_ENTRY, #4 IntegerUO schema gap (all applied); #5/#6/#7 pending |
| [43](#43-bccc-scsvul-2024-deep-dive-phase-1) | 2026-06-06 | BCCC-SCsVul-2024 Deep Dive — Phase 1 | 68,433 unique contracts (38.8% file dup), 41% multi-label, 12 classes (2 not in SENTINEL's 10), 92% pre-0.6 Solidity, CSV md5 verified |
| [44](#44-bccc-scsvul-2024-deep-dive-phase-2--final-cleaned-dataset) | 2026-06-06 | BCCC-SCsVul-2024 Deep Dive — Phase 2 | 8 workstreams complete; 67,311 contracts in 10-class SENTINEL v9 schema; 70/15/15 stratified split; 73% compile rate; 0 overlap with SmartBugs |
| [46](#46-bccc-deep-dive-phase-5--label-verification) | 2026-06-08 | **BCCC Deep Dive — Phase 5 (Label Verification)** | 46,977 labels dropped; 7,403 kept; 18,751 reclassified NonVulnerable; `contracts_clean_v1.3.csv` + `v1.4.csv` produced |

### 6. Pre-Flight Fixes (Pre-Run 5/7/8/9) & Audit

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [23 (line 1132)](#23-gnn-interpretability-suite) | 2026-05-30 | GNN Interpretability Suite | A1–A5, S1–S3 interpretability experiments |
| [24](#24-interpretability-suite--audit-fixes--phase-b-measurements) | 2026-06-01 | Interpretability — Audit Fixes + Phase B | EXP-L1–L4 reclassification; temperature calibration; 3 new scripts |
| [25](#25-run-5-pre-flight--phase-0-critical-safety-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 0 | A20 (label=0 hardcoded), A38 (NaN loss backward order) |
| [26](#26-run-5-pre-flight--phase-1-data--schema-layer-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 1 | A1–A3, NF-2, A19, A21, A22 (data + schema layer) |
| [27](#27-run-5-pre-flight--phase-2-graph-extraction-layer-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 2 | A4–A18, NF-1/7/10/11 (graph extraction layer) |
| [29](#29-run-5-pre-flight--phase-3-model-architecture-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 3 | A23, A25–A30, A33, A34, NF-6/8 (model architecture) |
| [30](#30-run-5-pre-flight--phase-4-training-loop-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 4 | A35–A37, NF-4, NF-9, StructuredLogger (training loop) |
| [31](#31-run-5-pre-flight--phase-5-training-interventions--cli-hardening) | 2026-06-02 | Run 5 Pre-Flight — Phase 5 | `aux_phase2_loss_weight` propagation, size-stratified F1, `--fusion-max-nodes` CLI |
| [32](#32-run-5-pre-flight--phase-46-training-log-spec-gap-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 4.6 | `training_logger.py` new/fixed; trainer wiring (StructuredLogger, gate metrics) |
| [33](#33-v9-findings-validation--code-fixes-c-1c-3h-2m-3m-6nf-6--run-5-kill) | 2026-06-02 | v9 Findings Validation + Code Fixes | C-1, C-3, H-2, M-3, M-6, NF-6 fixed; Run 5 killed |
| [36](#36-fix-35--safe-resume-rng-state--full-optimizer-restore) | 2026-06-04 | Fix #35 — Safe Resume | `resume_model_only` default False; 4 RNG streams + `tuned_thresholds` saved/restored |
| [37](#37-pre-run-9-audit-findings) | 2026-06-05 | Pre-Run 9 Audit Findings | 10 findings A–J; Run 8 test F1=0.2307; test contracts OOD; predictor hardcoded 0.55 tier |
| [38](#38-pre-run-9-fixes) | 2026-06-06 | Pre-Run 9 Fixes (also in §5) | #1–#8 fixes applied/pending; schema bump v8→v9 |
| [§45](#45-model-evaluation-dashboard-v2-spec-rewrite) | 2026-06-06 | Model Evaluation Dashboard v2 spec | v1 (2026-06-04) was 60-70% aligned; v2 rewrite addresses 14 gaps + 9 improvements; 39 reqs / 10 components / 27 properties / 7 phases / ~60 tasks |

### 7. ADRs (Architectural Decisions)

| § | Date | ADR | One-line summary |
|---|------|-----|------------------|
| [ADR-0001](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0001 — Schema versioning | `FEATURE_SCHEMA_VERSION` string, cache key suffix, module-level asserts |
| [ADR-0002](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0002 — Multi-label formulation | (also in §3) 10-class sigmoid multi-hot |
| [ADR-0003](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0003 — Dual-path Four-Eye architecture | (also in §3) GNN + LoRA-CodeBERT + 4 eyes |
| [ADR-0004](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0004 — Three-phase GAT routing | (also in §3) 8-layer, Ph1/Ph2/Ph3, sub-routing |
| [ADR-0005](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0005 — BCCC-SCsVul-2024 dataset | 41,576 contracts; SmartBugs held out as OOD; 87.9% pre-0.8 |
| [ADR-0006](#41-tier-1-architectural-decision-records) | 2026-06-06 | ADR-0006 — Loss formulation | (also in §3) ASL γ⁻=2 γ⁺=1 + aux BCE + JK entropy |

### 8. Agent Layer & MLOps

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [16](#16-three-tier-ml-output) | 2026-05-27 | Three-Tier ML Output | (also in §3) CONFIRMED/SUSPICIOUS/NOTEWORTHY schema |
| [17](#17-mlops--model-registry--drift-detector) | 2026-05-27 | MLOps — Model Registry + Drift Detector | `promote_model.py` gates; `exercise_drift_detector.py` CI test |
| [18](#18-agent-layer--three-tier-schema-integration) | 2026-05-27 – 05-28 | Agent Layer — Three-Tier Schema Integration | 78 tests pass; routing.py, nodes.py, inference_server.py, test_graph_routing.py |
| [19](#19-agent-layer--step-d-graph-inspector) | 2026-05-29 | Agent Layer — Step D: Graph Inspector | Phase 1 graph inspector implementation |
| [20](#20-agent-layer--step-e-cross_validator--graph-topology) | 2026-05-29 | Agent Layer — Step E: cross_validator + Graph Topology | Cross-validator + graph topology checks |
| [21](#21-agent-layer--phase-1-a1a2a3-hotspots--gnn-attention--quick_screen) | 2026-05-30 | Agent Layer — Phase 1 A1/A2/A3 | Hotspots, GNN attention, quick_screen |
| [22](#22-agent-layer--phase-1-a4a5-aderyn-deep-path--end-to-end-smoke-test) | 2026-05-30 | Agent Layer — Phase 1 A4/A5 | Aderyn deep-path, end-to-end smoke test |

### 9. Feature Specs & API Design

| § | Date | Title | One-line summary |
|---|------|-------|------------------|
| [§45](#45-model-evaluation-dashboard-v2-spec-rewrite) | 2026-06-06 | Model Evaluation Dashboard v2 spec | v1 (2026-06-04) was 60-70% aligned; v2 rewrite: 39 reqs / 10 components / 27 properties / 7 phases / ~60 tasks; new `ml/src/api/` module |

### Sequential Section Index (chronological)

| # | Date | Title |
|---|------|-------|
| [1](#1-project-foundation) | 2026-04-26 → 04-29 | Project Foundation |
| [2](#2-v4-baseline) | pre-2026-05-09 | v4 Baseline |
| [3](#3-v50--three-eye-architecture) | 2026-05-11 – 05-12 | v5.0 — Three-Eye Architecture |
| [4](#4-v51--dataset-deduplication) | 2026-05-12 | v5.1 — Dataset Deduplication |
| [5](#5-v52--jk--lora--three-phase-gnn) | 2026-05-14 – 05-16 | v5.2 — JK + LoRA + Three-Phase GNN |
| [6](#6-v53--asl-loss-experiment) | 2026-05-16, killed | v5.3 — ASL Loss Experiment |
| [7](#7-v6--graph-feature-schema-patch) | 2026-05-17 | v6 — Graph Feature Schema Patch |
| [8](#8-v7--full-architecture-overhaul) | 2026-05-18 – 05-19 | v7 — Full Architecture Overhaul |
| [9](#9-v8--cross-function-graph-extension) | 2026-05-19 – 05-21 | v8 — Cross-Function Graph Extension |
| [10](#10-v8-ab--joint-icfgdef_use-ablation) | 2026-05-20 | v8-AB — Joint ICFG+DEF_USE Ablation |
| [11](#11-plan-3a--icfg-only-ablation) | 2026-05-21 – 05-23 | PLAN-3A — ICFG-Only Ablation |
| [12](#12-phase-35--data-quality-fixes) | 2026-05-23 | Phase 3.5 — Data Quality Fixes |
| [13](#13-phase-36--graphcodebert--gnn-prefix-injection) | 2026-05-23 – 05-24 | Phase 3.6 — GraphCodeBERT + GNN Prefix Injection |
| [14](#14-imp--architectural-fixes--p1-train-run-2) | 2026-05-24 | IMP-* Architectural Fixes + P1-TRAIN Run 2 |
| [15](#15-p1-train-runs-3-and-4) | 2026-05-25 – 05-26 | P1-TRAIN Runs 3 and 4 |
| [16](#16-three-tier-ml-output) | 2026-05-27 | Three-Tier ML Output |
| [17](#17-mlops--model-registry--drift-detector) | 2026-05-27 | MLOps — Model Registry + Drift Detector |
| [18](#18-agent-layer--three-tier-schema-integration) | 2026-05-27 – 05-28 | Agent Layer — Three-Tier Schema Integration |
| [19](#19-agent-layer--step-d-graph-inspector) | 2026-05-29 | Agent Layer — Step D: Graph Inspector |
| [20](#20-agent-layer--step-e-cross_validator--graph-topology) | 2026-05-29 | Agent Layer — Step E: cross_validator + Graph Topology |
| [21](#21-agent-layer--phase-1-a1a2a3-hotspots--gnn-attention--quick_screen) | 2026-05-30 | Agent Layer — Phase 1 A1/A2/A3 |
| [22](#22-agent-layer--phase-1-a4a5-aderyn-deep-path--end-to-end-smoke-test) | 2026-05-30 | Agent Layer — Phase 1 A4/A5 |
| [23](#23-gnn-interpretability-suite) | 2026-05-30 | GNN Interpretability Suite |
| [23](#23-pre-run-5-implementation--label-cleaning--cei-aux-loss--calibration) | 2026-05-31 | Pre-Run-5 Implementation (label cleaning + CEI aux loss + calibration) |
| [24](#24-interpretability-suite--audit-fixes--phase-b-measurements) | 2026-06-01 | Interpretability Suite — Audit Fixes + Phase B Measurements |
| [25](#25-run-5-pre-flight--phase-0-critical-safety-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 0 |
| [26](#26-run-5-pre-flight--phase-1-data--schema-layer-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 1 |
| [27](#27-run-5-pre-flight--phase-2-graph-extraction-layer-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 2 |
| [28](#28-run-5-training-log-specification) | 2026-06-02 | Run 5 Training Log Specification |
| [29](#29-run-5-pre-flight--phase-3-model-architecture-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 3 |
| [30](#30-run-5-pre-flight--phase-4-training-loop-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 4 |
| [31](#31-run-5-pre-flight--phase-5-training-interventions--cli-hardening) | 2026-06-02 | Run 5 Pre-Flight — Phase 5 |
| [32](#32-run-5-pre-flight--phase-46-training-log-spec-gap-fixes) | 2026-06-02 | Run 5 Pre-Flight — Phase 4.6 |
| [33](#33-v9-findings-validation--code-fixes-c-1c-3h-2m-3m-6nf-6--run-5-kill) | 2026-06-02 | v9 Findings Validation + Code Fixes + Run 5 Kill |
| [34](#34-v10-re-extraction-launch--script-defaults-alignment) | 2026-06-02 | v10 Re-Extraction Launch + Script Defaults Alignment |
| [35](#35-run-7-architecture--issue-14-fixes) | 2026-06-03 | Run 7 Architecture + ISSUE-1–4 Fixes |
| [36](#36-fix-35--safe-resume-rng-state--full-optimizer-restore) | 2026-06-04 | Fix #35 — Safe Resume |
| [37](#37-pre-run-9-audit-findings) | 2026-06-05 | Pre-Run 9 Audit Findings |
| [38](#38-pre-run-9-fixes) | 2026-06-06 | Pre-Run 9 Fixes |
| [39](#39-v9-schema-upgrades) | 2026-06-06 | v9 Schema Upgrades |
| [40](#40-run-9-launch--watcher) | 2026-06-06 | Run 9 Launch + Watcher |
| [41](#41-tier-1-architectural-decision-records) | 2026-06-06 | Tier 1 Architectural Decision Records |
| [42](#42-run-9-v11--crash--resume--lambda-typo) | 2026-06-06 | Run 9 v11 — Crash + Resume + Lambda Typo |
| [43](#43-bccc-scsvul-2024-deep-dive-phase-1) | 2026-06-06 | BCCC-SCsVul-2024 Deep Dive — Phase 1 |
| [44](#44-bccc-scsvul-2024-deep-dive-phase-2--final-cleaned-dataset) | 2026-06-06 | BCCC-SCsVul-2024 Deep Dive — Phase 2 — Final Cleaned Dataset |
| [45](#45-model-evaluation-dashboard-v2-spec-rewrite) | 2026-06-06 | Model Evaluation Dashboard — v2 Spec Rewrite |

**Known numbering note (preserved, not renumbered):** §23 is duplicated in source (GNN Interpretability Suite at L1132 + Pre-Run-5 Implementation at L1195), and §28/§29 are physically out of order in source. Both preserved as-is to keep existing anchors stable.

---

## 1. Project Foundation

**Period:** 2026-04-26 – 2026-04-29
**Commits:** `9cb6990` → `8343d34`

### What was established

- **Architecture concept:** Dual-path GNN + CodeBERT multi-label vulnerability detector for Solidity smart contracts. GNN encodes contract structure (graph); CodeBERT encodes token semantics; CrossAttentionFusion combines both; three-eye classifier produces per-class logits.
- **ZK proof layer:** EZKL/Groth16 ZKML circuit proves model outputs on-chain. Proxy MLP (Linear 128→64→32→10) approximates the full model for ZK tractability.
- **Agent system:** LangGraph 5-agent topology — `ml_assessment → evidence_router → [rag_research ‖ static_analysis] → audit_check → synthesizer`. SqliteSaver checkpointing.
- **MCP servers:** Three SSE servers — `:8010` inference, `:8011` RAG, `:8012` audit.
- **Contracts:** Foundry layout under `contracts/src/` — `AuditRegistry.sol`, `SentinelToken.sol`, `IZKMLVerifier.sol`.
- **Initial bug fixes (2026-04-29):** predictor.py Bug 3/4/5/6/7; api.py missing torch import; vuln_type bridge via shared report store; faiss atomic write; rag_server lazy-load retriever.

### Infrastructure
- Python 3.12.1, Poetry, WSL2 Ubuntu
- RTX 3070 8GB VRAM
- MLflow with SQLite backend (`sqlite:///mlruns.db`)

---

## 2. v4 Baseline

**Period:** pre-2026-05-09
**Checkpoint:** `ml/checkpoints/multilabel-v4-finetune-lr1e4_best.pt`

### Architecture
- GNN + CodeBERT without CrossAttentionFusion Three-Eye design
- NODE_FEATURE_DIM: legacy (pre-v5)
- Dataset: 68,523-row `multilabel_index.csv` — **contained 34.9% cross-split content leakage** (same `.sol` file stored in multiple BCCC category directories; path-based MD5 created duplicate rows spanning splits)

### Results
- **Tuned F1-macro: 0.5422** — inflated by leakage; not a reliable number
- Best classes (tuned): IntegerUO=0.776 · GasException=0.507 · Reentrancy=0.519
- Behavioral test: not run

### Why these numbers are kept
v4 is retained as a **relative floor** only. v5+ must beat `v4 F1 − 0.05` per class on the clean dataset. The absolute 0.5422 is meaningless due to leakage.

---

## 3. v5.0 — Three-Eye Architecture

**Period:** 2026-05-11 – 2026-05-12
**Commits:** `80408c3` → `899adc9`

### Architecture changes
- Introduced **CrossAttentionFusion** — multi-head attention between GNN node embeddings and CodeBERT token embeddings (`attn_dim=256`, `output=128`)
- **Three-eye classifier** — three independent classification heads (GNN eye, Transformer eye, Fused eye) each contributing to the final loss
- GNN hidden_dim=128, `gnn_heads=8`, 4-layer GAT
- `NODE_FEATURE_DIM=12` (v3 schema — included `in_unchecked` feature, later found redundant)
- `NUM_EDGE_TYPES=7` (no REVERSE_CONTAINS yet)
- Dataset: still 68K leaky

### Results
- **Best val F1: 0.5828** (ep44) — **invalid due to 34.9% leakage**
- Behavioral test: **15% detection / 0% specificity** — model predicts "all-vulnerable" for any input

### Root causes of behavioral failure
- R1: `_select_contract()` interface bug selected wrong contract in multi-contract `.sol` files
- R2: ALL nodes pooled into GNN representation (should be FUNCTION nodes only)
- Dataset leakage meant model memorised cross-split duplicates rather than learning structural patterns

---

## 4. v5.1 — Dataset Deduplication

**Period:** 2026-05-12
**Commits:** `b12669c` → `f9d8f98`

### Data changes
- Implemented content-hash MD5 deduplication on raw Solidity source
- **Dataset reduced: 68,523 → 44,470 contracts**
- Splits regenerated: train=31,128 / val=6,236 / test=6,237 (approximate, based on deduped CSV)
- Leakage eliminated

### Training outcome
- Run `v5.1-fix28`: best F1=0.2794 (ep8) — training terminated early due to unresolved schema issues
- **Result declared INVALID** — schema v5 still had 12-dim features with `in_unchecked` bug

---

## 5. v5.2 — JK + LoRA + Three-Phase GNN

**Period:** 2026-05-14 – 2026-05-16
**Commits:** `b9ba690` (partial) → `46b216f`

### Architecture changes
- **JumpingKnowledge aggregation** (`jk_mode="attention"`) — learnable per-node aggregation across all GNN layers
- **LoRA** applied to CodeBERT Q+V projections — `r=16`, `α=32`, `dropout=0.1`, 589,824 trainable params from 124M frozen
- **Three-phase GNN:**
  - Phase 1 (layers 1–2): structural edges (CONTAINS, INHERITS, CALLS, READS, WRITES, EMITS)
  - Phase 2 (layers 3–5): control-flow edges (CONTROL_FLOW only)
  - Phase 3 (layers 6–7): REVERSE_CONTAINS (child→parent)
- `REVERSE_CONTAINS(7)` edge type added — runtime-only (built by dataset, not extractor)
- **AsymmetricLoss (ASL):** `γ_neg=2.0`, `γ_pos=1.0`, `clip=0.01` — replaced BCEWithLogitsLoss
- **Per-class pos_weight** with `pos_weight_min_samples=3000` cap
- **GNN gradient multiplier** (`gnn_lr_multiplier=2.5`) — counteracts GNN gradient collapse vs LoRA
- `torch.compile(model, dynamic=True)` enabled — ~20–40% step speedup

### Training runs
| Run | Best F1 | Tuned F1 | Notes |
|-----|---------|----------|-------|
| `v5.2-jk-20260515c` | 0.1872 (ep16) | — | Intermediate |
| `v5.2-jk-20260515c-r2` | 0.2823 (ep20) | 0.3373 | Early-stopped |
| `v5.2-jk-20260515c-r3` | **0.3306** (ep32) | **0.3422** | Best v5.2 |

### Behavioral failure analysis (v5.2-r3)
- Behavioral score: **36% detection / 33% specificity** — still far below usable
- Root cause: **~30% of test contracts were windowed from same source as training** — partial leakage from multi-window tokenization (same contract file → multiple windows, some in train, some in test). Fixed by stride-based deduplication in v6+.
- JK Phase 3 weight at convergence: 0.572→0.784 — model learned to almost entirely rely on REVERSE_CONTAINS, ignoring CFG signal. Structural ceiling confirmed.

---

## 6. v5.3 — ASL Loss Experiment

**Period:** 2026-05-16, killed
**Run:** `v5.3-bce-smooth-20260516`

- Switched back to BCE + label smoothing to test whether ASL γ was causing instability
- Best F1: 0.2559 (ep31) — **killed**, ASL confirmed superior
- No further investigation; v6 proceeded with ASL

---

## 7. v6 — Graph Feature Schema Patch

**Period:** 2026-05-17 – 2026-05-18
**Commits:** `8c8ce8c` → `b4ad806` (partial)

### What changed
- **In-place graph feature patch** applied to all 44,472 existing graphs — corrected NODE_FEATURE_DIM from 12→11 (removed redundant `in_unchecked` at feature index [9])
- `FEATURE_SCHEMA_VERSION = "v6"` — schema version tracking added
- Shape guard in `GNNEncoder.forward()` rejecting stale 12-dim files
- Assertion guards in `graph_schema.py` enforcing schema constant consistency at import time

### Why a patch not a re-extraction
Re-extraction would have taken 4+ hours. The `in_unchecked` feature was zeroed for all contracts (Slither doesn't emit it reliably), so patching in-place was byte-identical to re-extracting.

---

## 8. v7 — Full Architecture Overhaul

**Period:** 2026-05-18 – 2026-05-19 (training completed 2026-05-19)
**Commits:** `46a8f9d` → `47896a7`
**Training log:** `ml/logs/v7.0_best.log` · **Checkpoint:** `ml/checkpoints/v7.0_best.pt`

### What changed

#### Schema (v7)
- `NODE_FEATURE_DIM: 12 → 11` (permanent, `in_unchecked` removed as BUG-L2 fix)
- `FEATURE_SCHEMA_VERSION = "v7"`
- Full v7 re-extraction from all 44,470+ source contracts

#### 27 bugs fixed (across 4 phases)
Key fixes:

| Bug | Category | Fix |
|-----|----------|-----|
| BUG-L2 | Schema | `in_unchecked` removed; NODE_FEATURE_DIM 12→11 |
| BUG-H1 | GNN | `conv3c` (3rd CFG hop) added; `gnn_layers` 6→7 |
| BUG-H7 | Extractor | EMITS edges: EventCall IR fallback for Solidity 0.4.x |
| BUG-H8 | Extractor | INHERITS edges: parent CONTRACT nodes added before declaration |
| BUG-C3 | Extractor | CFG nodes inherit dims [1,3,4,5,9] from parent FUNCTION |
| BUG-M1 | Extractor | `return_ignored`: `id(lval)` → `lval.name` for stable hashing |
| BUG-M3 | Loss | ASL `γ_neg`/`γ_pos`/`clip` per-class tensors; registered as buffers |
| BUG-M8 | Eval | Per-epoch per-class threshold sweep; logs `val_f1_macro_tuned` |
| C2 | Training | DoS gradient no longer leaks through 3 aux heads |
| C4 | CLI | `--dos-loss-weight` flag added |
| F4 | CLI | `"positive"` added to `--weighted-sampler` choices |

#### Speed stack
- **SDPA** (Scaled Dot-Product Attention) activated on CodeBERT — `TransformerEncoder` falls back from FA2 (not supported by RoBERTa architecture) to SDPA
- **Flash Attention 2.8.3** installed via pre-built wheel (ready for decoder backbone swap)
- **Fused AdamW** (`fused=True`) — ~5% step speedup

#### DoS augmentation
- 60 contracts generated: 30 `dos_vuln_*.sol` + 30 `dos_safe_*.sol`
- 6 fail compilation (nested interface syntax — accepted, skipped)
- 110 augmented `.sol` files total in `ml/data/augmented/`

#### Dataset after v7 extraction
- **41,576 graphs** (schema v7, 11-dim)
- **44,470 token files** (`ml/data/tokens_windowed/` — `[4, 512]` per contract, stride=256)
- Splits (deduped): train=29,103 / val=6,236 / test=6,237
- Cache: `ml/data/cached_dataset_v7.pkl` (2.0 GB)

### Training results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | **0.2651** (ep23) |
| Tuned F1-macro | **0.2875** |
| Killed at | ep34, patience 10/30 |
| VRAM peak | 0.4 / 8.0 GiB |
| JK Phase 3 at plateau | 0.768 (Phase 3 dominance, Phase 2 = 0.182) |

**Top classes (tuned):** IntegerUO=0.706 · GasException=0.369 · Reentrancy=0.303
**Ceiling diagnosis:** JK Phase 2 decayed 0.35→0.18 over training. Model learned to distrust CFG signal because no cross-function edges exist — reentrant patterns spanning caller/callee are invisible.

### Behavioral tests
- 20 hand-crafted `.sol` contracts (`ml/scripts/test_contracts/`)
- 19 expected detections across 17 contracts (3 safe contracts, 1 contract with 3 expected)
- Baseline score: **8/19 (42%)** at tuned thresholds

---

## 9. v8 — Cross-Function Graph Extension

**Period:** 2026-05-19 – 2026-05-21
**Commits:** `519e3a8` → `5031a8a`
**Full doc:** `docs/changes/2026-05-19-v8-graph-extension-and-full-reextraction.md`

### Motivation
v7 ceiling (0.2875 tuned F1) confirmed architectural — no cross-function edges meant the GNN could not trace reentrancy patterns across caller/callee boundaries. Three new edge types implemented.

### New edge types

| Type | ID | Direction | Semantics |
|------|----|-----------|-----------|
| CALL_ENTRY | 8 | calling CFG node → callee entrypoint | Lets signal flow into callee body |
| RETURN_TO | 9 | callee terminal nodes → call-site successor | Post-call state flows back to caller |
| DEF_USE | 10 | def node → use node (intra-function) | Arithmetic def-use chains for IntegerUO |

**Schema:** `FEATURE_SCHEMA_VERSION = "v8"` · `NUM_EDGE_TYPES = 11` (was 8)

### Extractor changes
- `global_cfg_node_map` accumulated across all function iterations (was discarded per-function — PLAN-1C)
- `_add_icfg_edges()` — new helper for CALL_ENTRY(8) + RETURN_TO(9) cross-function edges
- `_add_def_use_edges()` — new helper for DEF_USE(10) intra-function data-flow edges
- `NodeType` IntEnum added to `graph_schema.py` (13 types — never hardcode raw integers)

### Full re-extraction
- **41,576 graphs extracted** (41,576 ok / 73 ghost / 2,875 skip / 0 fail — 29 min, 10 workers)
- Old v7 graphs archived to `ml/data/archive/graphs_v7/`
- Cache rebuilt: `ml/data/cached_dataset_v8.pkl` (2.2 GB, schema v8)

### Dataset statistics (v8)
```
Nodes: mean=125  P50=89   P99=623   max=1,735
Edges: mean=248  P50=145  P99=1,801 max=6,516

Edge type distribution (41,576 graphs):
  CALLS(0)          :    437,968
  READS(1)          :    641,801
  WRITES(2)         :    678,879
  EMITS(3)          :         12   (rare)
  INHERITS(4)       :    105,010
  CONTAINS(5)       :  3,672,916
  CONTROL_FLOW(6)   :  3,140,025
  CALL_ENTRY(8)     :    257,829   NEW — 63.7% of graphs
  RETURN_TO(9)      :    232,814   NEW — 55.5% of graphs
  DEF_USE(10)       :  1,159,688   NEW — 80.3% of graphs
  REVERSE_CONTAINS(7): runtime-only, added by dataset
```

### Label cleaning (v8 — first pass)
- `label_cleaner.py` re-run on v8 graphs
- 3,665 labels removed (structural mismatch: labels where graph features contradict vulnerability claim)

### Pre-training validation gates (all passed)
- GATE-3A-CACHE: cache size=2.16 GB, schema=v8, pairs=41,576
- GATE-3A-0: edge activation by class — CALL_ENTRY 63.7% overall, 68.3% on Reentrancy=1
- GATE-3A-1: edge mask code verification — DEF_USE(10) correctly excluded from PLAN-3A mask
- GATE-3A-2: config review (dos_weight, aug data, MLflow, sampler, log format)
- GATE-3A-VRAM: 8.0 GB free

---

## 10. v8-AB — Joint ICFG+DEF_USE Ablation

**Period:** 2026-05-20
**Run:** `v8-AB` · **Log:** `ml/logs/v8-AB_best.log`
**Full doc:** `docs/ml/v8-AB-training-analysis.md`

### Config
- Phase 2 edges: `CF(6) + CALL_ENTRY(8) + RETURN_TO(9) + DEF_USE(10)` — all new edge types
- `GNN hidden_dim=256`, `gnn_layers=7`, `gnn_heads=8`
- `LoRA r=16`, `α=32`
- `dos_loss_weight=0.0` (DoS gradient detached — too few samples at the time)

### Results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | 0.2621 (ep29) |
| Tuned F1-macro | **0.2851** |
| vs v7 tuned | −0.0024 |

**Key finding:** DEF_USE was diluting the Reentrancy signal. DEF_USE coverage on Reentrancy=1 contracts was 77.5% — the lowest of all classes — but CEI (Checks-Effects-Interactions) is a control-flow pattern, not a data-flow pattern. Adding DEF_USE added noise without adding relevant signal.

### JK collapse analysis
Per-node JK histogram run on 936 val contracts (123,139 nodes):

| Phase | Mean weight | Std | IQR | Dominance |
|-------|-------------|-----|-----|-----------|
| Phase 1 (structural) | 0.065 | 0.074 | 0.006–0.157 | 0.0% |
| Phase 2 (CF+ICFG+DFG) | 0.247 | 0.078 | 0.233–0.297 | 0.01% |
| Phase 3 (rev-CONTAINS) | 0.688 | 0.097 | 0.616–0.702 | 99.99% |

- Phase 2 IQR = 0.064 — global constant, no per-node routing
- Phase 3 dominates every node — JK has collapsed to a fixed weighting, not learned per-node routing
- **Not a regression vs v7** — same pattern observed; structural tendency of the architecture

**Full analysis:** `docs/ml/jk-attention-collapse-findings.md`

---

## 11. PLAN-3A — ICFG-Only Ablation

**Period:** 2026-05-21 – 2026-05-23
**Run:** `v8.0-A-20260521` · **Log:** `ml/logs/v8.0-A-20260521.log`
**Checkpoint:** `ml/checkpoints/v8.0-A-20260521_best.pt` · **Full doc:** `docs/ml/plan-3a-results.md`

### Config
- Phase 2 edges: `CF(6) + CALL_ENTRY(8) + RETURN_TO(9)` — **DEF_USE(10) dropped**
- All other config identical to v8-AB

### Results
| Metric | Value |
|--------|-------|
| Best raw F1-macro | **0.2790** (ep41) |
| Tuned F1-macro | **0.2877** |
| vs v7 tuned | +0.0002 (statistical tie) |
| vs v8-AB tuned | +0.0026 |
| Killed at | ep67 (patience 26/30) |

**Per-class winners vs v8-AB:** Timestamp +0.038, CallToUnknown +0.020, Reentrancy +0.005
**Per-class losers vs v8-AB:** IntegerUO −0.016, ExternalBug −0.015, TOD −0.011

**Key finding:** Timestamp was the biggest winner (+0.032 vs v7). DEF_USE had been *hurting* Timestamp — block-global assignments generate dense def-use chains, adding noise rather than signal. Dropping DEF_USE improved Timestamp dramatically and confirmed that DEF_USE placement (Phase 2) dilutes rather than enhances certain classes.

**Ceiling conclusion:** v7, v8-AB, PLAN-3A all converge to ~0.287 tuned F1. The ceiling is architectural — not data quality (see Phase 3.5 for the test of that hypothesis).

**Behavioral score:** Still ~8/19 — no improvement over v7.

---

## 12. Phase 3.5 — Data Quality Fixes

**Period:** 2026-05-23
**Commits:** `b9709f1` → `2b0cdfb` → `5a030d7`

### Motivation
Hypothesis H5: the ~0.287 ceiling is caused by label noise, not architecture. v8.0-B tests this by applying stricter label cleaning before training.

### DQ-1 — Stricter structural label cleaning

New precondition rules in `label_cleaner.py`:

| Class | Old rule | New rule | Labels removed |
|-------|----------|----------|----------------|
| Reentrancy | external_call_count > 0 | external_call_count > 0 **AND** has WRITES edge | −611 |
| Timestamp | uses_block_globals > 0 | uses_block_globals > 0 **AND** (external calls OR payable) | −568 |
| UnusedReturn | — | stricter return-value check | −1,665 |
| CallToUnknown | — | stricter external call pattern | −383 |
| MishandledException | — | stricter exception handling check | −632 |
| ExternalBug | — | structural check added | −445 |
| **Total** | | | **−4,304** |

Output: `ml/data/processed/multilabel_index_cleaned.csv`

### DQ-2 — DoS gradient re-enabled

- `dos_loss_weight`: 0.0 → **0.5** (fractional gradient scaling)
- Implementation upgraded from hard binary mask to true fractional scaling: `w * logit + (1−w) * logit.detach()`
- Trigger: ~243 DoS training positives available after augmentation (was 7 when originally disabled)

### DQ-3 — Complexity correlation diagnostic

`ml/scripts/complexity_correlation.py` — Spearman rank correlation between graph complexity metrics and predicted vulnerability probability across 1,500 val contracts (PLAN-3A checkpoint):

- Single alert: `ext_calls_sum` vs `MishandledException` r=0.402 (expected — ext calls are necessary)
- Highest near-shortcut: `num_nodes` vs `Timestamp` r=0.396 (BUG-H4 label noise)
- All other pairs: r=0.15–0.40 — moderate, no dominant shortcuts
- **Conclusion:** Model not dominated by raw size shortcuts. Label quality fixes expected to reduce these correlations.

### Bug fixes (adversarial audit, 2026-05-23)

Three real issues fixed (commits `0c650f1`, `5a030d7`):

| ID | Fix |
|----|-----|
| G-02 | `GNNEncoder.forward()`: mask operations on `edge_attr` are now in-place safe (no tensor aliasing) |
| G-03 | `select_prefix_nodes()`: tie-breaking in priority sort is deterministic (stable sort key) |
| T-04 | `TransformerEncoder._word_embeddings`: property accesses correct attribute path through LoRA wrapper |
| trainer | `._orig_mod.` prefix stripped from checkpoint state_dict at **save** time (not just load time) |

### v8.0-B training run — H5 refutation

**Run:** `v8.0-B-20260523` · **Config:** PLAN-3A edges + cleaned labels + `dos_loss_weight=0.5`

| Metric | Value |
|--------|-------|
| Best F1-macro | 0.2460 (ep10) |
| Killed at | ep11 |
| H5 verdict | **REFUTED** |

F1=0.2460 < 0.288 architectural ceiling. Data cleaning alone cannot break the ceiling. **The ceiling is purely architectural.** GraphCodeBERT + GNN prefix injection becomes the primary next intervention.

---

## 13. Phase 3.6 — GraphCodeBERT + GNN Prefix Injection

**Period:** 2026-05-23 – 2026-05-24 (ongoing)
**Proposal:** `docs/proposal/2026-05-23-graphcodebert-gnn-prefix-injection-proposal.md`
**Execution plan:** `docs/proposal/EXECUTION_PLAN.md`

### Core idea
Two simultaneous interventions:

1. **GraphCodeBERT drop-in:** Replace `microsoft/codebert-base` with `microsoft/graphcodebert-base` — same RoBERTa architecture and vocabulary, but pre-trained on code+data-flow graphs. Zero other changes for the P0 baseline.

2. **GNN prefix injection (Phase 1):** Project K=48 declaration-level GNN node embeddings [K,256] → [K,768] and prepend as prefix tokens to the transformer input. Uses `inputs_embeds` path instead of `input_ids`, making graph structure visible to **all 12 GraphCodeBERT attention layers** — not just at the final CrossAttentionFusion step.

### Prerequisites (all completed 2026-05-23)

| Check | Result |
|-------|--------|
| PRE-1 GraphCodeBERT downloaded | ✅ — 499MB model cached |
| PRE-2 Tokenizer identity (CodeBERT = GraphCodeBERT) | ✅ — vocab_size=50,265, token IDs identical on Solidity |
| PRE-3 UNK token rate | ✅ — **0.000000%** (0/418,248 tokens) |
| PRE-4 Node count distribution audit (K=48 covers 95.5%) | ✅ — declaration-level P95=47, K=48 chosen |
| PRE-5 LoRA applies to GraphCodeBERT | ✅ — 589,824 trainable params via `query`/`value` module names |

**Retokenization:** Not needed. TransformerEncoder truncates internally (`code_ids = input_ids[:, :code_budget]`). Existing stride=256 token files valid for K=48 (effective overlap = 208 tokens per window, no gap).

### P1-IMPL — Code implementation (completed 2026-05-23 – 2026-05-24)

| File | Change |
|------|--------|
| `ml/src/models/transformer_encoder.py` | `gnn_prefix_nodes: Optional[Tensor]` added to `forward()`; `inputs_embeds` path with prefix+code concatenation; position_ids: prefix=1 (RoBERTa padding slot), code=3..466; multi-window prefix expansion; `_word_embeddings` property; `WindowAttentionPooler.prefix_k` shifts CLS extraction right by K |
| `ml/src/models/sentinel_model.py` | `gnn_prefix_k`, `gnn_prefix_warmup_epochs`, `_current_epoch`; `gnn_to_bert_proj: Linear(256,768)`; `prefix_type_embedding: Embedding(5,768)`; `select_prefix_nodes()` with priority sort (CONSTRUCTOR > FALLBACK > RECEIVE > MODIFIER > FUNCTION); prefix suppressed during warmup (`gnn_prefix_nodes=None` when epoch < warmup_epochs) |
| `ml/src/preprocessing/graph_schema.py` | `NodeType` IntEnum (13 types); `STRUCTURAL_PREFIX_TYPES` frozenset (5 declaration types); `_PREFIX_NODE_PRIORITY`, `_PREFIX_TYPE_IDX` dicts |
| `ml/src/training/trainer.py` | `gnn_prefix_k`, `gnn_prefix_warmup_epochs`, `gnn_prefix_proj_lr_mult` in TrainConfig; separate `PrefixProj` param group (LR ×1.0); `model._current_epoch = epoch` each epoch; per-epoch `prefix_active` + `prefix_proj_weight_norm` logging to MLflow and logger |
| `ml/scripts/train.py` | `--gnn-prefix-k` (default 0), `--gnn-prefix-warmup-epochs` (default 15), `--gnn-prefix-proj-lr-mult` (default 1.0) |
| `ml/src/inference/predictor.py` | `gnn_prefix_k` and `gnn_prefix_warmup_epochs` passed from checkpoint config to SentinelModel; `model._current_epoch = 9999` after load (prefix always active at inference) |

**Key design decisions:**
- `prefix_type_embedding`: `Embedding(5, 768)` — 5 declaration types only (not 8; Phase 1 is declaration-level only)
- Warmup suppression: `gnn_to_bert_proj` receives **zero gradient** during warmup — not called at all. Starts from random init at ep16. Old plan (inject frozen prefix) would corrupt attention with untrained embeddings.
- No `--graphcodebert` CLI flag needed — TransformerEncoder is already hardcoded to graphcodebert-base after P0 change.
- Backward compatible: `gnn_prefix_k=0` (default) produces byte-identical output to original model.

### P0 — GraphCodeBERT drop-in results

**Run:** `graphcodebert-dropin-P0-20260523` (killed at ep4, best ep3)

| Epoch | F1-macro | Notes |
|-------|----------|-------|
| 1 | 0.1734 | GNN share 63–86%; JK Phase2=0.389 |
| 2 | 0.2056 | GNN share 67–75%; JK Phase2=0.398 |
| 3 | **0.2178** | JK Phase3=0.450; **ExternalBug=0.199, TOD=0.171** (both 0.000 in all CodeBERT runs) |

**GATE-GCB-2 verdict (2026-05-24): PASSED** — clear upward trend; ExternalBug and TOD both non-zero from ep3 confirming GraphCodeBERT cross-function pre-training signal.

### GATE-GCB-3 — Smoke test (2026-05-24, running now)

**Run:** `graphcodebert-v1-prefix48-smoke-20260524` (2 epochs, K=48)
**Log:** `ml/logs/graphcodebert-v1-prefix48-smoke-20260524.log`

Confirmed at startup:
- `GNN prefix K=48: WARMUP (starts ep15)` — warmup suppression active ✅
- `gnn_to_bert_proj weight norm: 16.0000` — proj at random init, not called ✅
- `PrefixProj=3 params (lr×1.0)` — separate param group registered ✅
- VRAM: 0.3/8.0 GiB — 7.7 GB free ✅

### P1-TRAIN — next step (blocked on GATE-GCB-3)

```bash
TRANSFORMERS_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_cache PYTHONPATH=. python ml/scripts/train.py \
    --run-name graphcodebert-v1-prefix48-$(date +%Y%m%d) \
    --experiment-name sentinel-gcb \
    --gnn-prefix-k 48 --gnn-prefix-warmup-epochs 15 --gnn-prefix-proj-lr-mult 1.0 \
    --phase2-edge-types 6 8 9 \
    --dos-loss-weight 0.5 \
    --cache-path ml/data/cached_dataset_v8.pkl \
    --splits-dir ml/data/splits/deduped \
    --label-csv ml/data/processed/multilabel_index_cleaned.csv \
    --weighted-sampler positive \
    --epochs 100 --early-stop-patience 30 --gradient-accumulation-steps 8
```

**Warmup schedule:** Epochs 1–14: prefix suppressed, model trains via CrossAttentionFusion path only. Epoch 15+: `gnn_to_bert_proj` starts from random init, K=48 GNN nodes injected as prefix tokens. Step loss expected to rise briefly at ep16 then recover.

**Target:** Tuned F1-macro > 0.30 (lifts architectural ceiling). Behavioral target: >10/19.

---

## Architecture Lineage Summary

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best tuned F1 |
|---------|-----------------|----------------|------------|----------|---------------|
| v4 | legacy | 7 | 4 | CodeBERT | 0.5422 (leaky) |
| v5.0 | 12 | 7 | 4 | CodeBERT | invalid (leaky) |
| v5.2 | 12 | 8 (+rev-CONTAINS) | 6 | CodeBERT+LoRA | 0.3422 (partial leak) |
| v6 | **11** | 8 | 6 | CodeBERT+LoRA | — (patch only) |
| **v7** | **11** | **8** | **7** | CodeBERT+LoRA | **0.2875** |
| v8-AB | 11 | **11** (+ICFG+DEF_USE) | 7 | CodeBERT+LoRA | 0.2851 |
| PLAN-3A | 11 | 11 (no DEF_USE in Ph2) | 7 | CodeBERT+LoRA | **0.2877** |
| v8.0-B | 11 | 11 | 7 | CodeBERT+LoRA | 0.2460 (data clean) |
| **GCB-P0** | 11 | 11 | 7 | **GraphCodeBERT**+LoRA | ~0.22 (3 ep) |
| GCB-P1-Run1 | 11 | 11 | 7 | GraphCodeBERT+LoRA+prefix K=48 | 0.2628 (ep27, killed ep28) |
| **GCB-P1-Run2** | 11 | 11 | **8** | GraphCodeBERT+LoRA+prefix K=48+**IMP-G1/G2/G3** | running |

**Architectural ceiling (all v7/v8 runs):** ~0.287 tuned F1. GraphCodeBERT + prefix injection is the intervention to break it.

---

## Open Bugs (as of 2026-05-24)

| ID | Impact | Status |
|----|--------|--------|
| BUG-H4 | Timestamp: ~463 contracts labeled positive with no timestamp features (inverted signal) | **DONE (2026-05-23)** — `label_cleaner.py` −568 Timestamp labels |
| BUG-H5 | Reentrancy: ~14% of positives have no external calls (structural impossibility) | **DONE (2026-05-23)** — `label_cleaner.py` −611 Reentrancy labels |
| BUG-M5 | Brainmab contract: standard ERC20 labeled Reentrancy+CallToUnknown+IntegerUO+MishandledException | OPEN |
| BUG-M6 | Token files carry stale `feature_schema_version='v4'` metadata | OPEN (auto-resolves on retokenize) |
| BUG-M7 | 8.5% of graphs have empty `contract_path` (can't cross-reference source) | OPEN |
| BUG-L3 | Hash-based graph-token pairing fragile to directory restructuring | DEFERRED |

---

## Key File Locations

| Asset | Path |
|-------|------|
| Best v7 checkpoint | `ml/checkpoints/v7.0_best.pt` (F1=0.2651) |
| Best PLAN-3A checkpoint | `ml/checkpoints/v8.0-A-20260521_best.pt` (tuned F1=0.2877) |
| GCB-P0 best checkpoint | `ml/checkpoints/graphcodebert-dropin-P0-20260523_best.pt` (F1=0.2178) |
| GCB-P1-Run1 best checkpoint | `ml/checkpoints/graphcodebert-v1-prefix48-20260524_best.pt` (F1=0.2628, ep27) |
| GCB-P1-Run2 log | `ml/logs/graphcodebert-p1-run2-20260524.log` (PID 80610, running) |
| v8 cache | `ml/data/cached_dataset_v8.pkl` (2.2 GB, 41,576 pairs) |
| Cleaned labels | `ml/data/processed/multilabel_index_cleaned.csv` |
| Splits | `ml/data/splits/deduped/` (train/val/test .npy files) |
| Behavioral test contracts | `ml/scripts/test_contracts/` (20 files, 19 expected detections) |
| MLflow DB | `mlruns.db` |
| Detailed training history | `docs/changes/` (session-level) · `docs/ml/` (analysis docs) |

---

## 14. IMP-* Architectural Fixes + P1-TRAIN Run 2

**Period:** 2026-05-24
**Session changelog:** `docs/changes/2026-05-24-imp-all-fixes-and-p1-train-run2.md`
**Analysis doc:** `docs/ml/gcb-p1-run1-analysis-and-imp-all.md`

### Motivation: P1-TRAIN Run 1 autopsy

Run 1 (7-layer GNN, K=48, killed ep28, best ep27 F1=0.2628) was killed after JK weight analysis revealed three root causes preventing the architectural ceiling from being broken:

| JK signal | Trajectory | Root cause |
|-----------|-----------|------------|
| Phase 1 (0.058–0.063) | **Flat throughout** | 11→256 dim change in conv1 discards all raw feature information without a skip connection |
| Phase 2 (0.387→0.234) | **Collapsing** | conv3/conv3b/conv3c shared identical `cfg_mask` edge set → identical inputs → JK correctly downweights them |
| Phase 3 (0.550→0.707) | **Growing unchecked** | REVERSE_CONTAINS became sole dominant signal; CFG nodes had no Phase 3 context → representation gap in CrossAttentionFusion |

Additional finding: `gnn_to_bert_proj` weight norm stagnated at 2 BF16 ULPs (16.0000→16.2500) across 13 post-warmup epochs — quantization floor at norm≈16 prevents fine-grained gradient accumulation.

### IMP-G1 — Phase 2 Layer-Specific Edge Subsets

**File:** `ml/src/models/gnn_encoder.py`

Phase 2 layers previously shared the full `cfg_mask` edge set (CF ∪ CALL_ENTRY ∪ RETURN_TO). Identical input + identical edge set = identical representation — JK attention correctly collapses them.

Fix: construct three distinct edge subsets before Phase 2:
- `cf_only_ei/ea` — `edge_attr == 6` (CONTROL_FLOW only)
- `icfg_only_ei/ea` — `edge_attr ∈ {8, 9}` (CALL_ENTRY + RETURN_TO, cross-function only)
- `cfg_ei/ea` — CF ∪ ICFG joint (existing mask)

Phase 2 routing: `conv3(cf_only)` → `conv3b(icfg_only)` → `conv3c(cfg joint)`

### IMP-G2 — Phase 1 Input Projection Skip

**File:** `ml/src/models/gnn_encoder.py`

New `__init__` parameter:
```python
self.input_proj = nn.Linear(NODE_FEATURE_DIM, hidden_dim, bias=False)  # 2,816 params
```

Forward block (before conv1):
```python
x_init = x
_proj_dtype = next(self.input_proj.parameters()).dtype
x_skip = self.input_proj(x_init.to(_proj_dtype)).to(x.dtype)
x = self.conv1(x_init, struct_ei, struct_ea)
x = self.relu(x + x_skip)
```

Also added dtype normalisation at `forward()` entry — required because GNN is always float32 while BERT load was polluting torch default dtype to BF16 (see DTYPE FIX below):
```python
_param_dtype = next(self.parameters()).dtype
if x.dtype != _param_dtype:
    x = x.to(_param_dtype)
```

### IMP-G3 — Phase 3 Bidirectional Context Pass

**File:** `ml/src/models/gnn_encoder.py`

Existing Phase 3 uses REVERSE_CONTAINS (CFG→FUNCTION upward). CFG nodes therefore received no Phase 3 context — only FUNCTION nodes were enriched by Phase 3. This created a representation gap that CrossAttentionFusion had to bridge across.

New downward CONTAINS pass (FUNCTION→CFG direction) after existing upward passes:
```python
self.conv4c = GATConv(hidden_dim, hidden_dim, heads=1, concat=False,
                      add_self_loops=False, edge_dim=_edge_dim)
# ...in forward():
x4c = self.conv4c(x, fwd_contains_ei, fwd_contains_ea)
x   = x + self.dropout(x4c)
x   = self.phase_norm[2](x)
```

Architecture is now **8 layers** (2+3+3). `gnn_num_layers` default updated 7→8 in `GNNEncoder.__init__`, `TrainConfig.gnn_layers`, and `trainer.py` warning threshold.

### IMP-M1 — FUNCTION Node Secondary Sort

**File:** `ml/src/models/sentinel_model.py`

`select_prefix_nodes()` now sorts FUNCTION nodes by `external_call_count` (feature[10]) descending when K truncation occurs. Sort key: `(priority, -ext_call_count, local_idx)`.

### IMP-M2 Tier 2 — prefix_attention_mean Diagnostic

**Files:** `ml/src/models/transformer_encoder.py`, `ml/src/training/trainer.py`

`TransformerEncoder.forward()` gains `gnn_prefix_counts` and `output_attentions` parameters. When `output_attentions=True`, extracts mean attention weight from code positions → prefix positions: `attn[:, :, :, K:, :K].mean()`.

New `SentinelModel.compute_prefix_attention_mean()` decorated `@torch.no_grad()`. Trainer logs `prefix_attention_mean` to MLflow each epoch post-warmup; warns if < 0.002.

### IMP-M3 — Zero-Padded Prefix Attention Mask Fix

**Files:** `ml/src/models/sentinel_model.py`, `ml/src/models/transformer_encoder.py`

`select_prefix_nodes()` return type changed to `tuple[Tensor, Tensor]`:
- `prefix` — `[B, K, 768]` projected node embeddings (zero-padded for graphs with < K nodes)
- `node_counts` — `[B]` int tensor, real node count per graph

`TransformerEncoder` constructs a count-based prefix attention mask so padded positions get `attention_mask=0` instead of attending to meaningless zero vectors.

### IMP-D1 — return_ignored Temporal Ordering Fix

**File:** `ml/src/preprocessing/graph_extractor.py`

`_compute_return_ignored()` rewritten. Old implementation built `all_read_names` as a global set across the entire function — false negative when a TemporaryVariable name collided with an unrelated read elsewhere.

New implementation: iterate `func.nodes` in CFG topological order, building `all_ops_ordered = [(node, op) for node in nodes for op in (node.irs or [])]`. For each call op at `call_idx`, check if `lval_name` appears in any `later_op.read` at `all_ops_ordered[call_idx + 1:]`. Uses direct `func.nodes` access (not `getattr`) so `AttributeError` propagates to the sentinel return.

Re-extraction of all 41K graphs pending (separate run, not blocking training).

### BF16 Global Dtype Side-Effect Fix

**File:** `ml/src/models/transformer_encoder.py`

`AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` calls `torch.set_default_dtype(bfloat16)` as a side effect, causing all `nn.Linear` created after BERT initialisation to have BF16 weights. Fixed:

```python
_prev_default_dtype = torch.get_default_dtype()
try:
    self.bert = AutoModel.from_pretrained(...)
finally:
    torch.set_default_dtype(_prev_default_dtype)
```

### Test suite (134/134 pass)

All 134 tests pass after the IMP-* changes. Key test fixes:

| Test file | Root cause | Fix |
|-----------|-----------|-----|
| `test_model.py` | `_StubTransformer` missing new params; `classifier` is now `Sequential` | Updated stub signature; `classifier[0].in_features`; shape `(N,128)→(N,256)` |
| `test_preprocessing.py` (schema) | v8 schema: `NODE_FEATURE_DIM=11`, `NUM_EDGE_TYPES=11`, `in_unchecked` removed | Updated all schema constant assertions |
| `test_preprocessing.py` (ReturnIgnored) | IMP-D1 API change: `func.slithir_operations` → `func.nodes[i].irs` | Tests now build `_make_mock_slither_node(irs=[...])` |
| `test_preprocessing.py` (integration) | `graph.x[:, 0]` stores `type_id / 12.0`; `.int()` always 0; raw integer comparisons invalid | Added `_type_ids()` / `_type_mask()` helpers using `(x * 12).round().long()` |
| `test_preprocessing.py` (integration) | Slither inserts intermediate CFG_NODE_OTHER between WRITE and CALL | Relaxed edge check to BFS reachability via `cf_adj` |
| `test_trainer.py` | Stale `scaler=scaler` kwarg removed from function signature | Removed from both `train_one_epoch()` calls |

### P1-TRAIN Run 2

**PID:** 80610  
**Log:** `ml/logs/graphcodebert-p1-run2-20260524.log`  
**Startup confirmed:** layers=8, gnn_prefix_k=48, warmup=15, VRAM 0.3/8.0 GiB, proj_norm=15.9853

Fresh start from random init (architecture changed — not resumed from Run 1). Target: tuned F1-macro > 0.30.

Monitor schedule:
- ep15: warmup ends
- ep16: prefix activates — expect brief loss spike then recovery
- ep17–20: `prefix_attention_mean` in MLflow; target > 0.005 by ep20
- ep17–20: Phase 2 JK weight — expect > 0.10 (vs 0.234 declining in Run 1)

---

## 15. P1-TRAIN Runs 3 and 4

**Period:** 2026-05-25 – 2026-05-26
**Commits:** `e022018`, `f1e228b`, `f2d0965`, `a6d4323`, `de4fe10`
**Log (Run 4):** `ml/logs/graphcodebert-p1-run4-20260525.log`
**Best checkpoint:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt`

### Run 2 outcome (killed 2026-05-25, ep4)

Run 2 died from JK Phase 3 weight spiking to 86.6% (structural collapse) — a different manifestation of the same IMP-G1 routing ambiguity. Autopsy confirmed ASL `pos_weight` passed through was amplifying DoS gradient ×20,000 combined with ASL gamma (double-amp, named NC-4).

### Run 3 — 8L + entropy reg λ=0.01 (killed ep16)

**Config:** 8-layer GNN, `jk_entropy_reg_lambda=0.01`, ASL pos_weight still passed (NC-4 not yet identified as root cause).

| Metric | Value |
|--------|-------|
| Best F1 | 0.17 (stuck) |
| Killed at | ep16 |
| Kill reason | NC-4 double-amp + JK uniform collapse (λ=0.01 forced 33/33/33 routing, no per-node differentiation) |

### Run 4 pre-flight fixes (commits `e022018`, `f1e228b`)

| Fix ID | Description |
|--------|-------------|
| NC-4 REVERTED | `pos_weight` NOT passed to ASL — double-amp was Run 3 primary root cause (DoS 10× pos_weight × ASL gamma ≈ 20,000× amplification) |
| C-3 λ 0.01→0.005 | Halved JK entropy regularisation — λ=0.01 forced perfect uniform 33/33/33 (no per-node routing); λ=0.005 allows differentiation |
| NC-1 logging fix | Reports total params vs state-initialised; clears all params regardless of count |
| C2 scatter collision fix | `_scatter_to_dense`: `valid = local_idx < max_nodes` evaluated BEFORE clamp — excess nodes truly dropped, not collided at slot 1023 |
| C1 gnn_enc_norm | `_grad_norm(model.gnn)` logged every step alongside `gnn_eye_proj` — full backbone visibility |
| H5 aux_fused compile | `aux_fused` added to `torch.compile` submodule list |

Speed logging commit `f2d0965` added per-epoch step timing to log output (applied at ep3, Run 4 restarted from ep3 checkpoint).

### Run 4 — 8L + λ=0.005 + no ASL pos_weight (best ep32, F1=0.3362)

**Run 4 PID:** 82628  
**Launched:** 2026-05-25 14:52 local, restarted 16:55 at ep3.  
**Killed:** 2026-05-26 20:12 at ep44 — F1 locked 0.31–0.34 for 12 epochs, capacity ceiling reached.

| Epoch | F1 | Notes |
|-------|-----|-------|
| ep6 | 0.2555 | First strong epoch |
| ep9 | 0.2670 | Run 1 beaten |
| ep10 | 0.2787 | New best |
| ep13 | 0.3153 | 0.30 barrier broken; DoS+Timestamp first detections |
| ep21 | 0.3224 | Plateau resolved; prefix working |
| ep25 | **0.3272** | New best at time |
| ep32 | **0.3362** | Final best checkpoint |
| ep44 | — | Killed — 12-epoch F1 ceiling 0.31–0.34 |

**ep30 gate (11:23 2026-05-26):** CONTINUE — best F1=0.3272 (+0.0395 above PLAN-3A tuned peak 0.2877); loss still declining; patience 5/30.

**Top classes (ep32):** IntegerUO=0.647, Timestamp=0.329, GasException=0.317  
**Hard classes (ep32):** CallToUnknown≈0.247, ExternalBug≈0.246, TOD≈0.235

**Pending fixes committed but requiring restart to take effect:**
- JK STD threshold 0.05→0.015 (commit `a6d4323`) — prevents false-alarm warnings
- `prefix_attention_mean` unpack bug: 3-value vs 4-value (commit `de4fe10`)

### Training history summary

| Run | Config | Best ep | F1 | Kill reason |
|-----|--------|---------|-----|-------------|
| GCB-P0 | GraphCodeBERT drop-in | ep3 | 0.2178 | GATE-GCB-2 passed |
| GCB-P1-Run1 | GCB+K=48 prefix (7L GNN) | ep27 | 0.2628 | IMP fixes needed |
| GCB-P1-Run2 | 8L GNN, all IMP | ep4 | — | JK Phase3=86.6% structural collapse |
| GCB-P1-Run3 | 8L+entropy reg λ=0.01 | ep16 | 0.17 (stuck) | NC-4 double-amp + JK uniform collapse |
| **GCB-P1-Run4** | **8L+λ=0.005+no ASL pw** | **ep32** | **0.3362** | Killed ep44 — capacity ceiling |

---

## 16. Three-Tier ML Output

**Period:** 2026-05-27

### `ml/src/inference/predictor.py`

Added three-tier threshold logic:
- `TIER_CONFIRMED_THRESHOLD = 0.55` — probability above which a class is "confirmed_vulnerable"
- `TIER_SUSPICIOUS_THRESHOLD = 0.25` — probability above which a class is "suspicious"
- New `_format_result()` method producing structured output with keys: `confirmed`, `suspicious`, `probabilities`, `tier_thresholds`

### `ml/src/inference/api.py`

- `VulnerabilityResult` gains `tier` field (`"safe"` | `"suspicious"` | `"confirmed_vulnerable"`)
- `PredictResponse` gains full three-tier schema: `confirmed`, `suspicious`, `probabilities` dict
- `/health` endpoint returns `tier_thresholds`, `model_epoch`, `model_f1_val`

### Label values

| Value | Meaning |
|-------|---------|
| `"safe"` | All class probabilities < `TIER_SUSPICIOUS_THRESHOLD` |
| `"suspicious"` | At least one class in `[TIER_SUSPICIOUS_THRESHOLD, TIER_CONFIRMED_THRESHOLD)` |
| `"confirmed_vulnerable"` | At least one class ≥ `TIER_CONFIRMED_THRESHOLD` |

Legacy label `"vulnerable"` still accepted by consumers for backward compatibility.

---

## 17. MLOps — Model Registry + Drift Detector

**Period:** 2026-05-27

### `ml/scripts/promote_model.py`

- Production F1 gate: new model must beat current production F1 before promotion
- Baseline check: rejects promotion if source stage is `"warmup"` (unless `--require-baseline` bypassed)
- Threshold JSON artifact logged to MLflow run at promotion time
- `--require-baseline` CLI arg: enforce warmup-stage source check

### `ml/scripts/exercise_drift_detector.py`

Four-phase smoke test script:
1. Warm-up: 500 requests to establish service baseline
2. Baseline build: `drift_detector.record_baseline()` called on in-distribution contracts
3. In-distribution check: detector should NOT fire on clean contracts
4. Shift detection: distributional shift injected; detector expected to fire

Uses `np.random.default_rng(42)` for reproducibility. Exits 0 on pass, 1 on failure.

---

## 18. Agent Layer — Three-Tier Schema Integration

**Period:** 2026-05-27 – 2026-05-28
**Commits:** `4aca6ff`

### Routing (`agents/src/orchestration/routing.py`)

Added `_iter_class_probs()` backward-compatibility helper. Old `ml_result` format used a flat list of `(class, prob)` tuples; new format uses a `probabilities` dict. Helper handles both transparently.

### Nodes (`agents/src/orchestration/nodes.py`)

Updated nodes to consume three-tier schema:

| Node | Change |
|------|--------|
| `evidence_router` | Reads `ml_result["probabilities"]` dict; falls back via `_iter_class_probs()` |
| `rag_research` | Builds query from confirmed + suspicious class names |
| `static_analysis` | Scopes Slither detectors to confirmed + suspicious classes |
| `synthesizer` | Reads `confirmed`/`suspicious` keys; formats tier into final report |

### State (`agents/src/orchestration/state.py`)

- `external_call_summary` field added to `AuditState` TypedDict
- Comment on `ml_result` updated to document three-tier schema keys

### ExternalBug structural gap fix

`_extract_external_call_summary()` function added using `fn.high_level_calls` from Slither. `rag_research` uses the summary to build a targeted query. `synthesizer` includes the call graph summary in the LLM prompt context.

### Tests

- `agents/tests/test_graph_routing.py` completely rewritten — prior version referenced non-existent fields from the old flat schema
- `agents/src/mcp/servers/inference_server.py`: `_mock_prediction()` rewritten to return full three-tier schema (`confirmed`, `suspicious`, `probabilities`, `tier_thresholds`)

---

## 19. Agent Layer — Step D: Graph Inspector

**Period:** 2026-05-29

### New MCP server (`agents/src/mcp/servers/graph_inspector_server.py`)

Port 8013, SSE transport. Phase 1 implementation (true GNN attention weights deferred to Phase 2).

**Tool:** `get_graph_hotspots(contract_code: str, flagged_classes: list[str]) → list[HotspotResult]`

**Scoring engine (`_analyze_hotspots()`):**

| Signal | Weight |
|--------|--------|
| Slither detector hit for class | ×3.0 |
| Structural signal match for class | ×1.0 |
| External-facing function | ×0.5 |
| Complexity (cyclomatic proxy) | ×0.2 |

**Key constants:**
- `_CLASS_STRUCTURAL_SIGNALS` dict: 10 vulnerability classes → list of structural feature names to check
- `_DETECTOR_CLASS_MAP` dict: Slither detector name → vulnerability class

Returns top-20 hotspot nodes ranked by score. Each result includes: `node_id`, `function_name`, `score`, `signals` (list of matched signals), `detector_hits` (list of matched Slither detectors).

### New `graph_explain` node (`agents/src/orchestration/nodes.py`)

Calls `sentinel-graph-inspector:get_graph_hotspots` via MCP. Always runs alongside `rag_research` + `static_analysis` in the deep path (fan-out of three).

State written:
- `ml_hotspots`: flat list of hotspot dicts (top-20 nodes, score, signals)
- `graph_explanations`: per-class breakdown dict + graph statistics (`node_count`, `edge_count`, `hotspot_count`)

---

## 20. Agent Layer — Step E: cross_validator + Graph Topology

**Period:** 2026-05-29

### New `cross_validator` node (`agents/src/orchestration/nodes.py`)

LLM-adjudicated per-class verdicts. Runs after `audit_check`, before `synthesizer` (deep path only).

**Verdict values:**
| Verdict | Meaning |
|---------|---------|
| `CONFIRMED` | All evidence streams agree — vulnerable |
| `LIKELY` | ML + at least one tool agree |
| `DISPUTED` | ML and static analysis contradict each other |
| `WATCH` | Suspicious ML signal, no static confirmation |
| `SAFE` | No evidence across all streams |

**Evidence context built per class:**
- ML tier and probability from `ml_result`
- Slither findings filtered to class from `static_analysis` output
- RAG topics mentioning class from `rag_research` output
- Prior audit count for class from `audit_check` output

Falls back to empty dict on LLM failure — `synthesizer` uses rule-based fallback in that case.

### `synthesizer` update

When `cross_validator` verdicts are present in state, `synthesizer` uses them directly rather than rebuilding evidence assessment. Falls back to rule-based logic when verdicts dict is absent or empty.

### Graph topology update (`agents/src/orchestration/graph.py`)

New topology:

```
START → ml_assessment → evidence_router
         ↓ (deep path only)
  [rag_research ‖ static_analysis ‖ graph_explain]
         ↓
  audit_check → cross_validator → synthesizer → END
         ↓ (shallow path)
  synthesizer → END
```

`evidence_router` fans out to three parallel nodes on the deep path. `cross_validator` is inserted after `audit_check`.

### Test suite

23 new tests added:
- `graph_explain` node: hotspot retrieval, MCP error handling, empty result handling
- `cross_validator` node: all 5 verdict types, LLM failure fallback, partial class coverage
- Routing: fan-out-of-three verification, shallow path unchanged
- `build_graph`: topology structure assertions, node/edge count checks

**Total test count: 187 passing.**

### Architecture lineage update

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best F1 |
|---------|-----------------|----------------|------------|----------|---------|
| v7 | 11 | 8 | 7 | CodeBERT+LoRA | 0.2875 tuned |
| v8-AB | 11 | 11 | 7 | CodeBERT+LoRA | 0.2851 tuned |
| PLAN-3A | 11 | 11 | 7 | CodeBERT+LoRA | 0.2877 tuned |
| GCB-P0 | 11 | 11 | 7 | GraphCodeBERT+LoRA | 0.2178 (3 ep) |
| GCB-P1-Run1 | 11 | 11 | 7 | GraphCodeBERT+LoRA+prefix K=48 | 0.2628 (ep27) |

---

## 21. Agent Layer — Phase 1 A1/A2/A3: Hotspots + GNN Attention + quick_screen

**Period:** 2026-05-30
**Commits:** `4bf5ba5` (A1) → `9ede400` (A2) → `94ca2c5` (A3)

### Motivation

Three gaps identified in the existing agent layer needed addressing before Run 5 data quality work:

1. **GNN attention was fake** — graph_inspector_server used Slither as a proxy for GNN attention hotspots. The real signal was in the model's GNN node embeddings, unused.
2. **ML blind spot** — contracts where all class probabilities fell below DEEP_THRESHOLDS were routed directly to the synthesizer with zero static tool evidence. A contract consistently scoring below 0.25 on all classes (plausible for obfuscated or atypical contracts) would receive a "safe" report based on ML alone.
3. **graph_inspector_server Phase 2** needed A1 to land first.

### A1 — `/hotspots` ML Inference Endpoint

**Files:** `ml/src/inference/predictor.py`, `ml/src/inference/api.py`, `ml/tests/test_api.py`

**What was added:**

`predict_with_hotspots(source_code: str) -> dict` method in `Predictor`:
- Runs the same windowed preprocessing as `predict_source`
- Calls `GNNEncoder.forward()` to get per-node embeddings `x` of shape `[N, 256]`
- Filters to function-type nodes (`_FUNC_TYPE_IDS_SET = {1,2,4,5,6}` — FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR)
- Computes L2 norm per node → normalized `[0,1]` score
- Maps node index → function name via `graph.node_metadata[i]["name"]`
- Returns full ML result + `"hotspots"` list + `"hotspot_stats"` dict

`POST /hotspots` FastAPI endpoint:
- Request: `{"source_code": str}`  (same Pydantic validator as `/predict`)
- Response: `HotspotsResponse` — `{hotspots, hotspot_stats, label, probabilities, confirmed, suspicious}`
- Top-20 functions sorted by embedding-norm score descending
- One round-trip: full ML result + hotspots together

9 new tests in `TestHotspotsEndpoint` — all 18 API tests pass.

### A2 — graph_inspector_server Phase 2 (Real GNN Attention)

**File:** `agents/src/mcp/servers/graph_inspector_server.py`

**What changed:**

Phase 2 fallback chain: `_analyze_hotspots_gnn()` → `_analyze_hotspots_slither()` → `_mock_hotspots()`

`_analyze_hotspots_gnn()`: calls `POST {_ML_API_URL}/hotspots` via httpx, transforms response into the inspector's hotspot format. Returns `None` on any failure (timeout, HTTP error, malformed response) so the fallback chain activates.

Health endpoint now reports `phase: "2"` and lists active backends.

Config: `_ML_API_URL = os.getenv("SENTINEL_ML_API_URL", "http://localhost:8000")`, `_HOTSPOTS_TIMEOUT = 60`.

### A3 — quick_screen Tier-0 Node

**Files:** `agents/src/orchestration/state.py`, `agents/src/orchestration/nodes.py`, `agents/src/orchestration/graph.py`, `agents/tests/test_graph_routing.py`

**What changed:**

New `quick_screen` LangGraph node inserted between `ml_assessment` and `evidence_router`:

```
ml_assessment → quick_screen → evidence_router → [fan-out or fast path]
```

The node runs on **every** contract (Tier 0), regardless of ML score. It:
- Runs Slither with `_SCREEN_SLITHER_DETECTORS` (15 High-impact detectors)
- Runs Aderyn subprocess; parses JSON output for H-*/C-* rule IDs
- Both tools are non-fatal — ImportError/FileNotFoundError/subprocess failure all yield empty list
- Returns `{"quick_screen_hits": {"slither": [...], "aderyn": [...]}}`

`_route_from_evidence_router` updated with two-signal gate:
- Fast path: ML safe AND screen clean → synthesizer
- Screen-escalated deep path: ML safe BUT screen fired → `["static_analysis", "graph_explain"]`
- Normal deep path: ML flagged → existing fan-out unchanged

`evidence_router` node updated to log quick_screen signal in `routing_decisions`.

`AuditState` updated: `quick_screen_hits: dict[str, list[str]]` field added.

**Tests:** 16 new tests — `TestQuickScreenNode` (7), routing escalation cases (5), graph node set check updated. 59/59 passing in `test_graph_routing.py`, 212/212 across all agent tests.

### Why the design choices

- **In-process** (not MCP): Slither and Aderyn are installed in the process. MCP hop for Tier-0 screening would add latency with no benefit — same rationale as `static_analysis` (established pattern in existing code).
- **High/Critical only**: Running all 90+ Slither detectors in quick_screen would produce too many informational hits and escalate clean contracts. Only High-impact detectors (reentrancy-eth, arbitrary-send-eth, etc.) justify escalation.
- **Non-fatal design**: quick_screen is a screening gate, not a hard dependency. Slither unavailable → skip silently. Aderyn not installed → skip silently. Node always returns both keys regardless.
- **Separate node not merged with static_analysis**: quick_screen is scoped differently (broader detector set, both tools) and runs before routing. `static_analysis` runs after routing (deep path only) with class-scoped detectors. Merging them would couple the Tier-0 screen to the deep-path tool.

| Version | NODE_FEATURE_DIM | NUM_EDGE_TYPES | GNN layers | Backbone | Best F1 |
|---------|-----------------|----------------|------------|----------|---------|
| GCB-P1-Run4 | 11 | 11 | **8** | GraphCodeBERT+LoRA+prefix K=48+IMP | **0.3362** (ep32) |

---

## 22. Agent Layer — Phase 1 A4/A5: Aderyn deep-path + End-to-End Smoke Test

**Period:** 2026-05-30
**Commit:** `330e68e`

### A4 — Aderyn in deep-path static_analysis

**File:** `agents/src/orchestration/nodes.py`

**Why:** quick_screen runs Aderyn at Tier 0 (High/Critical only, Tier-0 screen). The deep path should also benefit from Aderyn's full detector set, giving cross_validator and synthesizer a second independent signal alongside Slither.

**Note on scope correction:** The original A4 spec said "add Aderyn to audit_server MCP (:8012)". During implementation we clarified that audit_server is the on-chain AuditRegistry server, not a static analysis tool server. The correct home for Aderyn is in-process in `static_analysis`, consistent with how Slither already runs. Result is equivalent: deep path now runs both tools.

**What was added:**

`_run_aderyn_on_file(tmp_path)` helper function:
- Runs `aderyn --output json <file>` as a subprocess (90s timeout)
- Parses JSON output: iterates `high`, `medium`, `low` buckets
- Extracts `id/title`, `description`, `instances` (line numbers + function names)
- Returns findings with `tool="aderyn"`, `impact` derived from bucket name, `confidence="Medium"`
- Non-fatal: `FileNotFoundError`, `TimeoutExpired`, any other exception → returns `[]`

Integration into `static_analysis`:
- Called on the same `tmp_path` after Slither finishes (file still exists before `finally` cleanup)
- `findings.extend(aderyn_findings)` — combined into the single `static_findings` list
- Log line updated: `slither=N aderyn=M external_calls=K`

**Impact on cross_validator:** the `static_findings` list now has both `tool="slither"` and `tool="aderyn"` entries. The existing `compute_verdict()` in routing.py looks for Slither detector names — it ignores Aderyn findings (they use rule IDs like "H-1" not Slither detector names). This is intentional: Aderyn findings are available to the synthesizer LLM but don't auto-confirm ML verdicts without a routing.py mapping. Adding ADERYN_TO_CLASSES mapping is a Phase 2 item (B1 in the plan).

### A5 — End-to-End Smoke Test

**File:** `agents/tests/test_smoke_e2e.py` (7 tests, all pass)

**Coverage:**

| Test | What it verifies |
|------|-----------------|
| `test_deep_path_vault_produces_final_report` | Full Phase 1 field set in final_report; path_taken=="deep" |
| `test_quick_screen_hits_in_final_state` | quick_screen_hits present with slither+aderyn keys |
| `test_deep_path_graph_explanations_present` | graph_explanations non-empty after deep path |
| `test_routing_decisions_logged` | routing_decisions populated with class names |
| `test_fast_path_safe_contract` | ML safe + screen clean → path_taken=="fast", empty rag_evidence |
| `test_screen_escalated_path_when_ml_safe_but_screen_fires` | ML safe but Slither fires → NOT fast path |
| `test_ml_failure_still_produces_report` | Inference server down → still produces report with ML failure note |

All MCP calls are mocked. Slither runs in-process on the Vault contract for the screen-escalated test (using `patch.object(slither_module, "Slither", ...)` to inject a mock instance). Aderyn subprocess is stubbed with `FileNotFoundError` to avoid CI dependency.

---

## 23. GNN Interpretability Suite (2026-05-30)

**Period:** 2026-05-30
**Commits:** pending
**Checkpoint evaluated:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (ep32, F1=0.3362)

### Overview

A structured GNN interpretability and validation suite was designed, implemented, run, and documented against the Run 4 best checkpoint. The study spans three analysis layers: (1) structural data quality, (2) architectural expressivity, and (3) learned model behaviour. 21 experiments were designed; 22 scripts were written; 12 Solidity test contracts were written. Experiments running against the checkpoint were executed; those requiring `solc` were blocked by a toolchain gap.

### Scripts and Experiments

| Location | Count | Description |
|----------|-------|-------------|
| `ml/scripts/interpretability/` | 22 scripts | One per experiment plus `utils.py` and `__init__.py` |
| `ml/scripts/interpretability/test_contracts/` | 12 files | Solidity contracts for EXP-L6 counterfactual testing |
| `docs/interpretability/` | 24 files | One markdown report per experiment + master report + index |

Experiments by layer:

| Layer | Experiments | Scripts |
|-------|-------------|---------|
| 1 — Structural | EXP-S1, S2, S3, S4, A1, A2 | exp_s1 through exp_s4, exp_a1, exp_a2 |
| 2 — Expressivity | EXP-E1, E2, E3, E4 | exp_e1 through exp_e4 |
| 3 — Learning | EXP-A3, A4, L1–L10 | exp_a3, exp_a4, exp_l1 through exp_l10 |

Additionally: 3 validation scripts (`val_finding1_jk_weights.py`, `val_finding2_proper_ablation.py`, `val_finding4_timestamp_size.py`) and a training ablation shell script (`run_training_ablation.sh`).

### Key Findings (validated)

| Finding | Experiment | Key Number | Status |
|---------|-----------|-----------|--------|
| Phase 3 (REVERSE_CONTAINS) marginally dominant in JK weights | EXP-L1, EXP-A3 | Phase 3 = 0.346–0.381, entropy 99.98% of max | Validated: all phases nearly equally weighted; regulariser prevents collapse but not Phase 2 suppression |
| CFG edge ablation has near-zero effect | EXP-L2 | CONTROL_FLOW+CALL_ENTRY combined Δ = 1.08×10⁻⁶ | Validated: five orders below the 0.03 threshold |
| GNN eye useful for only 3/10 classes | EXP-A4 | CallToUnknown, IntegerUO, Timestamp only | Validated: GNN F1=0 for 7 classes including Reentrancy |
| All 4 tested classes WL-distinguishable | EXP-E2 | Timestamp 0% collision, Reentrancy 11.1% | Validated: expressivity is not the bottleneck |
| All classes severely miscalibrated | EXP-L7 | ECE range 0.205–0.310, mean ≈ 0.252 | Validated: calibration required before deployment |
| Timestamp size shortcut confirmed at prediction level | EXP-L7 | F1=1.0 (small/medium) vs F1=0.364 (large); gap=0.636 | Validated: Cohen's d=1.657 (EXP-S3) now confirmed in predictions |
| `type_id_norm` dominates permutation importance by 3× | EXP-L8 | Mean importance 0.0786 vs next 0.0262 | Validated: `uses_block_globals` rank 10 (not last — stale feature names caused wrong earlier claim) |
| CEI paths present in data but GNN doesn't traverse them | EXP-S4 + EXP-L2 | 76% of Reentrancy-pos have CALL_ENTRY; ablation effect = 5.3×10⁻⁷ | Validated: data gap is not the cause |
| EMITS edges near-absent (12 total across 41K contracts) | EXP-S2 | 12 edges, baseline 0.051% | Validated: not extractable as signal |
| CEI reachability only 37.7% at k=8 | EXP-E1 | 37.7% Reentrancy-positive vs 26.7% negative | Validated: GNN hop depth insufficient for most CEI paths |

### Script Bugs Fixed

| Script | Bug | Fix |
|--------|-----|-----|
| `exp_l8_permutation_importance.py` | Stale `FEATURE_NAMES` list from pre-v8 schema (11 entries in wrong order) | Now imports `FEATURE_NAMES` from `graph_schema.py` |
| `exp_l1_jk_weight_analysis.py` | `mean_entropy` field stored mean JK weight (0.333) instead of Shannon entropy (1.099) | Fixed field extraction to compute entropy from per-phase weights |
| `ml/src/models/gnn_encoder.py` | Docstring said forward returns 3-tuple; `return_intermediates=True` returns 4-tuple | Docstring updated to document both return shapes |

### Output Files

| Path | Description |
|------|-------------|
| `docs/interpretability/INTERPRETABILITY_MASTER_REPORT.md` | Full 970-line report with per-class analysis, root cause analysis, capacity ceiling analysis, and Run 5 recommendations |
| `docs/interpretability/EXPERIMENT_INDEX.md` | Experiment table (22 rows), root cause analysis for failures, argparse bugs fixed |
| `docs/interpretability/exp_*.md` | 22 individual experiment reports with method, results, pass/fail, and implications |
| `docs/proposal/GNN_INTERPRETABILITY_FIXES_PROPOSAL.md` | Concrete fix proposals for Run 5 based on validated findings |


---

## 23. Pre-Run-5 Implementation — Label Cleaning + CEI Aux Loss + Calibration (2026-05-31)

**Period:** 2026-05-31
**What changed:** All code changes required before Run 5 training implemented.

### Interpretability Script Fixes (Phase A)

| Script | Fix |
|--------|-----|
| `exp_l4_gradient_saliency.py` | Import FEATURE_NAMES from graph_schema (stale pre-v8 list replaced) |
| `exp_a2_cfg_inheritance.py` | `_CONTAINS_EDGE=0` → `EDGE_TYPES["CONTAINS"]` (was matching CALLS not CONTAINS) |
| `exp_e1_receptive_field.py` | Analysis 2/3 redesigned: REVERSE_CONTAINS is runtime-only; redesigned as CONTAINS + CALLS checks |
| `exp_l3_attention_visualization.py` | Extended to hook conv3b (CALL_ENTRY+RETURN_TO) alongside conv3 (CF) |

### New Measurement Scripts (Phase B)

| Script | Purpose |
|--------|---------|
| `exp_b1_phase2_gradient_norm.py` | Grad norm at each phase LayerNorm — measures Phase 2 gradient flow |
| `exp_b2_per_eye_ece.py` | Per-eye (GNN/TF/Fused) ECE — prerequisite for temperature scaling |
| `exp_b3_jk_weight_distribution.py` | JK weight distribution per class — tracks Phase 2 contribution |
| `exp_b4_unusedreturn_saliency.py` | Gradient saliency for UnusedReturn top-scored contracts |

### Calibration (Phase C)

- `ml/scripts/calibrate_temperature.py` — fits per-class temperatures via LBFGS minimising BCE NLL on val set. Outputs `ml/calibration/temperatures.json`. Run after B2.

### Data Quality Scripts (Phase D)

Scripts written but require manual validation before running. Reliability varies:

| Script | Sol ID | Reliability | Notes |
|--------|--------|-------------|-------|
| `ml/scripts/clean_integeruo_labels.py` | Sol-2 | **High** | Version detection is deterministic; skips contracts without source path |
| `ml/scripts/gate_timestamp_labels.py` | Sol-3 | **Medium** | Catches direct timestamp-in-branch patterns; misses indirect propagation through state variables |
| `ml/scripts/clean_reentrancy_labels.py` | Sol-1 | **Low** | CFG BFS misses cross-function reentrancy and is only as complete as the extractor's `CFG_NODE_WRITE` coverage. Run `--dry-run` and cross-check a sample with `slither --detect reentrancy-eth` before committing. Alternative: skip and rely on E1 CEI aux loss instead. |
| `ml/scripts/inject_openzeppelin_negatives.py` | IMP-D2 | **High** | OZ contracts are audited clean; adds genuine zero-label negatives |

**Execution results (2026-05-31 dry-runs):** All three label scripts produced results that make running them unsafe or redundant:

| Script | Dry-run result | Decision |
|--------|---------------|---------|
| Sol-2 IntegerUO | 0 removals — BCCC dataset is 100% pre-0.8.0 Solidity (0.4.x/0.5.x) | SKIP — labels correct |
| Sol-3 Timestamp | 380/380 (100%) removal rate — `uses_block_globals` (x[:,2]) is function-level only, never set on CFG_NODE_CHECK child nodes → false detection | SKIP — feature architecture incompatible |
| Sol-1 Reentrancy | 2,606/3,887 (67%) removal rate — too aggressive for LOW-reliability BFS heuristic | SKIP — rely on E1 CEI aux loss instead |

**Label CSV unchanged:** `ml/data/processed/multilabel_index_cleaned.csv` is the Run 5 input.

### Run 5 Training Changes (Phase E)

**E1 — CEI auxiliary loss on Phase 2 embeddings:**
- `gnn_encoder.py`: added `return_phase2_embs=True` mode returning gradient-attached Phase 2 tensor
- `sentinel_model.py`: added `aux_phase2` head (256→128→10, GELU+Dropout)
- `trainer.py`: `TrainConfig.aux_phase2_loss_weight=0.10`; CEI classes (ExternalBug/Reentrancy/TOD) get 3× weight

**E2 — Timestamp size normalisation:**
- `graph_extractor.py`: complexity (dim 5) normalised by contract total node count (E2 Interp-3 fix)
- `trainer.py`: `use_weighted_sampler="timestamp-size"` mode — large Timestamp+ contracts get 4× weight

**IMP-D1 readiness:**
- `sentinel_model.py`: `fusion_max_nodes` parameter wired through to CrossAttentionFusion
- `trainer.py`: `TrainConfig.fusion_max_nodes=1024`; raise to 2048 after re-extraction

---

## 24. Interpretability Suite — Audit Fixes + Phase B Measurements (2026-06-01)

**Period:** 2026-06-01  
**Checkpoint evaluated:** `ml/checkpoints/GCB-P1-Run4-no-asl-pw_best.pt` (ep32, F1=0.3362)  
**All 21 interpretability experiments + B1–B4 resolved. No pending experiments.**

### Audit Fixes Applied (2026-06-01)

Four corrections applied after re-examination of measurement methodology and source code cross-referencing:

#### EXP-B1 — Gradient Method Corrected (BCEWithLogitsLoss)

**Script:** `ml/scripts/interpretability/exp_b1_phase2_gradient_norm.py`

Original run backpropagated through raw logit `logits[0, class_idx]`. This is non-standard — training uses `BCEWithLogitsLoss` which applies sigmoid before gradient. Fixed to use `F.binary_cross_entropy_with_logits(logits[0, class_idx].unsqueeze(0), target=1)`.

Corrected P2/P1 ratios: **72–91%** (was 75–86% with raw logit). Ordering unchanged. Timestamp remains highest (91.3%), DenialOfService lowest (72.2%).

| Class | Phase1 | Phase2 | Phase3 | P2/P1 |
|-------|--------|--------|--------|-------|
| CallToUnknown | 0.051893 | 0.041190 | 0.035657 | 79.4% |
| DenialOfService | 0.034882 | 0.025193 | 0.020473 | 72.2% |
| ExternalBug | 0.044038 | 0.033292 | 0.028976 | 75.6% |
| GasException | 0.041313 | 0.032616 | 0.025441 | 78.9% |
| IntegerUO | 0.039092 | 0.028423 | 0.022379 | 72.7% |
| MishandledException | 0.045841 | 0.036230 | 0.031026 | 79.0% |
| Reentrancy | 0.050614 | 0.037378 | 0.035040 | 73.8% |
| Timestamp | 0.094683 | 0.086430 | 0.073734 | 91.3% |
| TOD | 0.033661 | 0.025190 | 0.020505 | 74.8% |
| UnusedReturn | 0.080223 | 0.061586 | 0.052428 | 76.8% |

#### EXP-L2 — Structural Ablation Added

**Script:** `ml/scripts/interpretability/exp_l2_edge_ablation.py`  
**Output:** `ml/interpretability_results/exp_l2/exp_l2_ablation_delta.json`

Prior report only measured embedding ablation (zeroing edge embeddings). Structural ablation (removing edges entirely from the graph) was missing, inflating the denominator of the embedding/structural ratio. Structural ablation rerun and added.

Results:

| Metric | Value |
|--------|-------|
| cfg_combined_drop_embedding | 1.11×10⁻⁶ |
| cfg_combined_drop_structural | 0.0121 |
| Ratio (structural / embedding) | **10,944×** (corrected from earlier claimed 450×) |

Key sign-reversal finding: removing Phase 2 edges (CONTROL_FLOW, CALL_ENTRY, RETURN_TO) **increases** Reentrancy prediction scores:

| Edge type | Reentrancy structural delta | Timestamp structural delta |
|-----------|---------------------------|--------------------------|
| CONTROL_FLOW | +0.020328 | +0.162891 |
| CALL_ENTRY | +0.010085 | +0.019450 |
| RETURN_TO | +0.018186 | ≈0 |

Phase 2 is actively suppressing Reentrancy predictions — a shortcut learned from the training distribution (dense CFG → large well-engineered → not Reentrancy).

#### EXP-L3 — Reclassified ARCHITECTURAL N/A

**Script:** `ml/scripts/interpretability/exp_l3_attention_visualization.py`

Prior report claimed PASS (100% CF fraction in top attention edges). Retracted: 100% CF fraction is architecturally guaranteed because conv3 is wired exclusively to the CF-only subgraph. This is not a learned finding.

Real finding from the corrected run (conv3b also hooked): **all GAT attention weights = 1.0 (uniform)**. No selective attention learned within CFG or ICFG edges. Phase 2 attention is a weighted average with weight 1.0 — equivalent to a simple sum.

Status: **ARCHITECTURAL N/A** (not PASS, not FAIL — criterion was ill-posed).

#### EXP-L4 — Rerun with Correct Feature Names

**Script:** `ml/scripts/interpretability/exp_l4_gradient_saliency.py`  
**Output:** `ml/interpretability_results/exp_l4/`

FEATURE_NAMES was a stale hardcoded pre-v8 list. Fixed to import from `graph_schema.py`. Rerun 2026-06-01. Corrected results confirmed original finding structure but with accurate feature labels.

Per-class top-3 (correct feature names):

| Class | Rank 1 | Rank 2 | Rank 3 |
|-------|--------|--------|--------|
| All 10 classes | `external_call_count` (21–24%) | `complexity` (10–11%) | varies |

Pass criteria:
- Timestamp `uses_block_globals` ≥ 20%: actual 10.0% → **FAIL**
- Reentrancy CFG_NODE_CALL + has_state_write ≥ 20%: actual 8.9% → **FAIL**
- Global sensitivity artifact confirmed: `external_call_count` dominates gradient for ALL classes regardless of class semantics.

### Phase B — New Measurement Scripts (All Complete)

| Script | Status | Key Finding |
|--------|--------|-------------|
| `exp_b1_phase2_gradient_norm.py` | COMPLETE | Phase 1 > Phase 2 > Phase 3; P2/P1 = 72–91% |
| `exp_b2_per_eye_ece.py` | COMPLETE | GNN/TF/Fused ECE 0.057–0.065 (good); main head ECE 0.249 (severe) |
| `exp_b3_jk_weight_distribution.py` | COMPLETE | Universal Phase3 > Phase1 > Phase2; no class upweights Phase 2 |
| `exp_b4_unusedreturn_saliency.py` | COMPLETE | external_call_count + complexity dominate; return_ignored rank 4 (2.3% diff only) |

### Temperature Calibration

Fitted `ml/calibration/temperatures_run4.json` via `ml/scripts/calibrate_temperature.py` (LBFGS on val set).

| Before | After |
|--------|-------|
| ECE 0.249 (main head, severely miscalibrated) | ECE 0.028 (post temperature scaling) |

### Phase G — Interpretability Completeness Gaps (All Resolved)

Six completeness gaps identified from the INTERPRETABILITY_AUDIT_AND_COMPLETENESS.md document, all resolved by 2026-06-01:

| Gap | Fix |
|-----|-----|
| P1 — EXP-S3 "dead feature" finding | Retracted — was CFG-node artifact; FUNCTION-node mean computed correctly |
| P2 — EXP-E1 DEF_USE missing from Phase 2 | DEF_USE(10) added; Phase 2 k=8 reachability now 38.2% (was 37.7%) |
| P3 — EXP-L5 wrong pooling | max+mean [512] pooling fixed; IntegerUO Phase1 F1 corrected 0.114→0.419 |
| P4 — EXP-E4 only CF tested | All 4 Phase 2 edge types tested (CF/DEF_USE/CALL_ENTRY/RETURN_TO); all 0.0% directional diff |
| P5 — EXP-L9 non-discriminative criterion | Relative-rank criterion; FAIL confirmed (safe CW > vuln CW, delta=−0.00654) |
| P6 — Documentation stale | EXPERIMENT_INDEX + MASTER_REPORT fully updated; 4 B-experiment docs created |

### Pre-Run-5 Root Cause Analysis

Full Phase 2 root cause analysis documented in `docs/pre-run-fixes/phase2_root_cause_analysis.md`. Seven confirmed root causes identified:

1. FUNCTION nodes get identity transform from Phase 2 (CF edges don't reach FUNCTION nodes)
2. `aux_phase2_loss_weight` = 0.0 throughout all of Run 4 (feature not yet added to TrainConfig)
3. Phase 2 has 8× lower attention head capacity than Phase 1 (heads=1 vs heads=8)
4. JK entropy regularizer pushes Phase 2 weight below 1/3 by default
5. DEF_USE edges get only 1 hop (Layer 5 only)
6. Phase 3 does Phase 2's job via REVERSE_CONTAINS lift
7. Suppression encoded in learned weights: dense CFG → not Reentrancy (inversion)

### Current State After This Section

- All 21 interpretability experiments resolved (+ B1–B4)
- Temperature calibration fitted
- Run 5 config committed (aux_phase2_loss_weight=0.10, timestamp-size sampler)
- Pre-run-fixes analysis docs in `docs/pre-run-fixes/`
- **Next action: Implement Run 5 pre-flight fixes (Phase 0 through 5)**

---

## 25. Run 5 Pre-Flight — Phase 0 Critical Safety Fixes (2026-06-02)

**Source:** `docs/pre-run-fixes/SENTINEL-Run5-Actionable Implementation Plan.md` — Phase 0

These two fixes are mandatory blockers: either can permanently corrupt training data or optimizer state, making all downstream work meaningless if left unfixed.

### A20 — `label=0` Hardcoded in Batch Extraction Fixed

**File:** `ml/src/data_extraction/ast_extractor.py`

**Problem:** The multiprocessing worker was constructed with `partial(self.contract_to_pyg, ..., label=0)`, hardcoding `label=0` in every extracted `graph.y` regardless of the true vulnerability label. Any code path loading graphs in binary mode (without label_csv) would receive all-zero labels, poisoning the data.

**Fix:**
- Added module-level `_labeled_pool_worker(path_label_pair, extractor, solc_binary, solc_version)` — picklable, receives `(contract_path, label)` tuples.
- Added `_LABEL_COLUMNS` constant listing the 10 vulnerability class columns in order.
- Added `label_csv: Optional[Path]` parameter to `extract_batch_with_checkpoint`.
- Before the version-group loop: loads label CSV into `label_map: Dict[str, List[int]]` keyed by md5_stem (= `get_contract_hash(contract_path)`).
- Per version group: builds `path_label_pairs = [(path, label_map.get(hash))]` and asserts all have labels (Gate 0.1 check).
- `contract_to_pyg` signature updated from `label: int = 0` to `label: Union[List[int], int, None] = None`.
  - `List[int]` → `float32` tensor `[NUM_CLASSES]` (multi-label, used by DualPathDataset multi-label mode).
  - `int` → `long` tensor `[1]` (legacy binary mode).
  - `None` → `long` zeros `[1]` with verbose warning.
- CLI: `--label-csv` argument added (default: `ml/data/processed/multilabel_index_cleaned.csv`).

**Gate 0.1 assertion:** `assert len(missing) == 0` fires before any `pool.imap` call if any contract is absent from the label_map, preventing silent label poisoning.

### A38 — NaN Loss `backward()` Ran Before NaN Check Fixed

**File:** `ml/src/training/trainer.py`

**Problem:** `torch.isfinite(loss)` was checked on line 713, **after** `loss.backward()` on line 650. Any NaN/Inf loss caused corrupted gradients to flow through `optimizer.step()`, poisoning Adam's running mean/variance estimates (`m_t`, `v_t`). These estimates are never reset, so the corruption persists for the rest of training.

**Fix:**
- Moved `loss_for_log = loss.item() * accum_steps` and `isfinite` check to immediately after loss computation — before any backward/step.
- On NaN/Inf: `optimizer.zero_grad(set_to_none=True)` zeros stale gradients from earlier micro-batches, then `continue` skips `backward()`, `optimizer.step()`, and `total_loss` accumulation.
- **Post-clip guard:** After `clip_grad_norm_()`, checks all trainable params for non-finite gradients (`_has_bad_grads`). If found (BF16 overflow despite finite loss), zeroes grads and skips `optimizer.step()` without incrementing `optimizer_step` counter. Logs a WARNING.
- `optimizer.step()` / `scheduler.step()` now only called in the `else` branch (clean gradients confirmed).
- Epoch-end summary updated: reports `nan_loss_count/{n_batches}` as a rate; logs `logger.error` if rate exceeds Gate 0.2 threshold (0.5%), instructing operator to halt training and restart from the last clean checkpoint.

### Gate Status After Phase 0

| Gate | Condition | Status |
|------|-----------|--------|
| 0.1 (DATA POISONING BLOCK) | `label_map` populated before pool; assert covers all contracts | **Code complete — verify after test extraction** |
| 0.2 (ADAM STATE CORRUPTION BLOCK) | NaN check precedes `backward()`; `nan_loss_count` logged per epoch | **Code complete — verify with smoke test (5 injected NaN steps)** |

---

## 26. Run 5 Pre-Flight — Phase 1 Data & Schema Layer Fixes (2026-06-02)

**Source:** `docs/pre-run-fixes/SENTINEL-Run5-Actionable Implementation Plan.md` — Phase 1

### A1 — Missing `max(NODE_TYPES)` Range Guard Fixed

**File:** `ml/src/preprocessing/graph_schema.py`

Added `assert max(NODE_TYPES.values()) == 12` after the existing `len(NODE_TYPES) == 13` check. The new guard fires at import time if a future schema addition changes the normalization divisor used throughout the pipeline, directing the developer to update every division-by-12 site before re-extracting.

### A2 — Uppercase Hex Accepted in Hash Validation Fixed

**File:** `ml/src/utils/hash_utils.py`

- Added `import re`.
- Replaced `int(hash_string, 16)` with `re.fullmatch(r'[0-9a-f]{32}', hash_string) is not None` in `validate_hash()`.
- Added explicit type and length checks before the regex so error messages are specific.
- Old behaviour silently accepted uppercase hex strings (e.g. `"ABCD..."`), which could allow two logically different identifiers to both pass validation.

### A3 + A32 — Dynamic `_MAX_TYPE_ID` Assertions Added

**Files:** `ml/src/preprocessing/graph_extractor.py`, `ml/src/models/sentinel_model.py`

Both files already derived `_MAX_TYPE_ID = float(max(NODE_TYPES.values()))` dynamically. Added `assert _MAX_TYPE_ID == 12.0` in both, with a message pointing to every normalization site that must be updated if a new node type is ever appended. The assertion fires at import, making schema drift a hard failure rather than a silent misalignment.

### NF-2 — Decode-Side Hardcoded `* 12` Fixed

**File:** `ml/src/preprocessing/graph_extractor.py` (`_add_node` ~L1094)

Replaced `int(round(x_list[-1][0] * 12))` with `int(round(x_list[-1][0] * _MAX_TYPE_ID))`. This is the mirror of A3/A32: the encode side used `_MAX_TYPE_ID` dynamically but the decode side was hardcoded to 12, creating a latent divergence bug if the schema ever gained a new node type.

### A19 — Solc Binary Resolution Uses `get_project_root()` Instead of `Path.cwd()`

**File:** `ml/src/data_extraction/ast_extractor.py`

- Moved `get_project_root()` definition before `get_solc_binary()` (clear dependency order).
- Replaced `Path.cwd() / ".venv" / ...` with `get_project_root() / ".venv" / ...` in `get_solc_binary()`.
- Old behaviour: invoking the extractor from any directory other than the repo root silently produced a wrong (non-existent) binary path, causing all contracts to fall back to the system solc without a version warning.

### A21 — Worker `print()` Under Concurrency Fixed

**File:** `ml/src/data_extraction/ast_extractor.py`

- Added `import logging` and `logger = logging.getLogger(__name__)`.
- Replaced all `print()` calls inside exception handlers (`GraphExtractionError`, bare `Exception`, no-label warning) with `logger.warning(...)` / `logger.error(...)`.
- `print()` is not concurrency-safe under multiprocessing — concurrent calls interleave on stdout. `logging.QueueHandler` (already used by workers) serialises output correctly.

### A22 — `torch.save` Without Error Handling Fixed

**File:** `ml/src/data_extraction/ast_extractor.py`

- Wrapped `torch.save(result, graph_file)` in `try/except (OSError, IOError)`.
- On failure: logs `logger.error(...)`, appends path to `failed_saves`, skips the `processed_hashes.add()` (contract is not marked done so it can be retried with `--resume`), and `continue`s the loop.
- After the version-group batch: if `failed_saves` is non-empty, raises a `RuntimeError` listing all failed paths so the caller surface them.
- Checkpoint verbose `print()` converted to `logger.info()` (A21 consistency).

### Gate 1.1 — TOOLCHAIN CHECK: PASSED

- `solc`: present in venv at `ml/.venv/bin/solc`, version 0.8.20
- `solc-select`: present in venv, **97 versions** available (covers 0.4.0–0.8.35)

---

## 27. Run 5 Pre-Flight — Phase 2 Graph Extraction Layer Fixes (2026-06-02)

**File:** `ml/src/preprocessing/graph_extractor.py` (all changes in this section)
**Plan ref:** §6 of `SENTINEL_Run5_UNIFIED_PREFLIGHT_PROPOSAL.md`

All graph_extractor.py bugs fixed before Phase 7 re-extraction. Corrected graphs will differ from the v8 dataset — re-extraction is required (Phase 7 covers this).

### A4 — `assert` Used for Production Invariant (node_metadata alignment)

- Replaced `assert len(node_metadata) == x.shape[0]` with `if … raise ValueError(…)`.
- Python `-O` (optimised mode) strips `assert` statements; alignment violations would silently produce misindexed metadata in production extraction runs.

### A5 — `except AttributeError` Scope Too Broad in `_compute_return_ignored`

- Narrowed the `try` block to only the `func.nodes` Slither API access.
- `from slither... import` moved to a separate `try/except ImportError` block.
- All remaining logic (the ordered-IR scan) is now outside any try, so unexpected `AttributeError`s propagate instead of silently returning `-1.0`.

### A6 — Bare `except Exception: pass` in `_compute_call_target_typed`

- Added `logger.debug("[A6] ...")` and incremented module-level `_call_target_fail_count`.
- Previously silent type-resolution failures masked Slither API drift.

### A7 — Dead Code `_compute_in_unchecked`

- Replaced the entire function body with `raise NotImplementedError(...)`.
- `in_unchecked` was dropped from the v7 feature vector (BUG-L2). Any surviving call site is now a loud, immediate error rather than a silent wrong-feature calculation.

### A8 — `is True` Identity Check in `_compute_has_loop`

- `getattr(func, "is_loop_present", None) is True` → `bool(getattr(func, "is_loop_present", False))`.
- `is True` rejects truthy non-bool values (e.g. integer `1`), silently returning `0.0` for loop-containing functions in some Slither versions.

### A9 — String-Based Class Check for `SolidityVariableComposed`

- Added module-level `from slither.core.declarations.solidity_variables import SolidityVariableComposed as _SolidityVariableComposed`.
- Import path confirmed working: `slither.core.declarations.solidity_variables`.
- Replaced `type(rv).__name__ == "SolidityVariableComposed"` with `isinstance(rv, _SolidityVariableComposed)`.
- If the import fails at module load, a prominent `WARNING` is logged and `uses_block_globals` always returns `0.0` (Timestamp/TOD detection severely degraded).

### A10 — Bare `except Exception: pass` in `_cfg_node_type`

- Added `logger.warning("[A10] ...")` and incremented module-level `_cfg_type_fallback_count`.
- High fallback counts in logs indicate Slither version mismatch (Gate 2.1 threshold: < 1% of CFG nodes).

### A11 — Hardcoded Parent Feature Indices in `_build_cfg_node_features`

- Added `_FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}` at module level.
- Replaced raw magic numbers `p[1]`, `p[3]`, `p[4]`, `p[5]`, `p[9]` with named lookups (`_FEAT_IDX["visibility"]`, `_FEAT_IDX["view"]`, etc.).
- Index drift on a future schema change no longer silently corrupts CFG node features.

### A12 — `n.node_id` Without Fallback in Sort Key

- `n.node_id` → `getattr(n, "node_id", 0)` in the `_build_control_flow_edges` sort key.
- Synthetic Slither nodes (e.g. ENTRY_POINT) may lack `node_id`; the old code raised `AttributeError`, aborting CFG extraction for the entire function.

### A13 — Dropped CONTROL_FLOW Edges Not Logged

- Added `else` branch in Pass 2 of `_build_control_flow_edges`: increments `_dropped_cf_edges` counter and logs at `DEBUG`.
- Per-function summary log at `DEBUG` level after the pass.

### A14 — RETURN_TO Cartesian Product Includes Revert Paths

- `_func_terminal_map` construction now filters out `THROW` and `RETURN` Slither node types before adding to the terminal set.
- Only normal-exit terminals (fall-through with no successors) produce `RETURN_TO` edges — revert/unwind paths must not connect to call-site successors.

### A16 — `assert` Used for Sentinel-Range Check in `_build_node_features`

- Replaced both `assert return_ignored in (-1.0, 0.0, 1.0)` and `assert call_target_typed in (-1.0, 0.0, 1.0)` with `if … raise ValueError(…)`.

### A17 — Exception Routing by String Keyword Matching

- Added type-based check via `from slither.exceptions import SlitherError` before string keyword matching in the Slither instantiation `except Exception` block.
- String matching falls back for CryticCompile errors; TODO comment added to replace with `isinstance` checks once `crytic_compile.errors` is confirmed stable.

### A18 — `except Exception: pass` in ICFG Map Construction

- Combined with A14 fix: replaced `except Exception: pass` with `except Exception as exc: _icfg_failure_count += 1; logger.error("[A18] ...")`.
- `_icfg_failure_count` is a local variable summed per contract; Gate 2.1 (`== 0`) is checked via log output after Phase 7 re-extraction.

### NF-1 — EMITS Edge Key Mismatch in Fallback Path

- Built `_event_name_map: dict[str, str]` from `contract.events` before the function-edges loop: maps short name (e.g. `"Transfer"`) to canonical name (e.g. `"ERC20.Transfer(address,address,uint256)"`).
- In the EventCall IR fallback, translated `ir.name` through `_event_name_map` before adding to `emitted` set.
- For Solidity <0.4.21 contracts, this repairs the fallback path that previously produced only 12 EMITS edges across 41K contracts (NF-1 confirmed root cause).

### NF-7 — Silent 0.0 Return on Failure

- `_compute_external_call_count`: `except Exception: return 0.0` → `except Exception as exc: _ext_call_fail_count += 1; logger.debug("[NF-7] ...")`.
- `_compute_uses_block_globals`: `except Exception: pass` → `except Exception as exc: _block_globals_fail_count += 1; logger.debug("[NF-7] ...")`.
- Both counters are module-level; logged in per-contract extraction summary.

### NF-10 — Duplicate Function Name

- Changed `for func in contract.functions:` to `for func_index, func in enumerate(contract.functions):`.
- `_add_node` signature extended with `override_key: str | None = None` parameter.
- When `_add_node` returns `None` (duplicate), now re-attempts with synthetic key `"{canonical_name}__override__{func_index}"` instead of reusing the first function's node index.
- Preserves overriding function's CFG (critical — overrides often introduce vulnerabilities).
- If synthetic key also collides (degenerate case), logs a WARNING and skips.
- `_duplicate_func_count` local counter logged per contract.

### NF-11 — `_add_edge` Drops ALL Edge Types Silently

- Added `_edge_drop_counts: dict[int, int] = {}` in outer scope before `_add_edge` definition.
- `_add_edge` now increments `_edge_drop_counts[etype]` and logs at `DEBUG` when either endpoint is missing.
- After INHERITS loop: per-type drop summary logged at `DEBUG` (e.g. `"CALLS: 3, READS: 1"`).
- Note: CONTROL_FLOW drops are separately addressed by A13.

### Module-Level Additions

- Added `_FEAT_IDX` (A11), `_SolidityVariableComposed` (A9), and four fail counters (`_call_target_fail_count`, `_cfg_type_fallback_count`, `_ext_call_fail_count`, `_block_globals_fail_count`).
- Import line extended to include `FEATURE_NAMES` from `graph_schema`.

### Gate 2.1 — EXTRACTION HEALTH (checked at Phase 7 re-extraction)

- `_icfg_failure_count == 0` — verified via log output after full re-extraction
- `_cfg_type_fallback_count / total_cfg_nodes < 0.01` — verified via log output
- CALL_ENTRY edge presence rate ≥ 64.2% — verified post-extraction
- RETURN_TO edge presence rate ≥ 55.6% — verified post-extraction

### Gate 2.2 — FEATURE VALIDATION (checked at Phase 7 re-extraction)

- `uses_block_globals` non-zero for ≥ 80% of Timestamp-positive contracts in val split — verified post-extraction

**Last updated: 2026-06-02**

---

## 29. Run 5 Pre-Flight — Phase 3 Model Architecture Fixes (2026-06-02)

**Files:** `ml/src/models/gnn_encoder.py`, `ml/src/models/transformer_encoder.py`, `ml/src/models/sentinel_model.py`
**Plan ref:** Phase 3 of `SENTINEL-Run5-Actionable Implementation Plan.md`

All fixes target correctness and performance of the model forward pass. No training dynamics change.

### A27 — `num_layers` Enforcement + `SENTINEL_GNN_NUM_LAYERS` Constant

**File:** `gnn_encoder.py`

Added module-level `SENTINEL_GNN_NUM_LAYERS: int = 8`. `GNNEncoder.__init__` now raises `ValueError` if `num_layers != 8` — the three-phase architecture is fixed at 8 layers (2+3+3) and any other value produces a structurally incorrect model. Previously the parameter was stored but never validated.

### A23 — `last_weight_stds` NaN for N=1

**File:** `gnn_encoder.py` — `_JKAttention.forward()`

Replaced `.std(0)` (uses `unbiased=True` by default — returns NaN when N=1, the single-sample diagnostic case) with `.std(0, unbiased=False).nan_to_num(0.0)`. The comment "0 if N=1" was wrong; this fix makes it true.

### A25 — `edge_index.max()` O(E) Scan on Every Forward Pass

**File:** `gnn_encoder.py`

Added `validate_graph_integrity: bool = False` parameter to `GNNEncoder.__init__`. The `edge_index.max()` integrity check (O(E) scan) is now gated behind this flag. Default is `False` in production — enable during debugging or testing. The check should be moved to `DualPathDataset.__getitem__` or the collation function for production-grade validation without per-forward overhead.

### A26 — `next(self.parameters())` Called Twice Per Forward Pass

**File:** `gnn_encoder.py`

Cached parameter dtype as `self._param_dtype` at the end of `__init__`. Both forward-pass dtype checks now use the cached value. Added `refresh_dtype_cache()` method — callers must invoke it after any runtime dtype cast (`.float()`, `.half()`, `.bfloat16()`) to keep the cache consistent.

### NF-6 (DEFERRED) — Phase 2 Layers 3/4 Ignore `phase2_edge_types` Ablation

Added inline code comment at the cf_only/icfg_only computation block explaining the bug and the fix. Implementation deferred to Run 6 per the Known Non-Fixes list — zero training impact in normal runs where `phase2_edge_types=None`.

### A28 — `except (ImportError, ValueError)` Catches Real BERT Load Errors

**File:** `transformer_encoder.py`

Narrowed `except (ImportError, ValueError)` to `except ImportError` only. Flash Attention 2 not being installed → `ImportError` → SDPA fallback. A `ValueError` from a corrupted `config.json` or missing model file is a real error and must propagate — previously it was silently swallowed into the SDPA fallback.

### A29 — Python Loop for Prefix Mask Construction (Two Occurrences)

**File:** `transformer_encoder.py`

Replaced `for b in range(B): prefix_mask[b, :gnn_prefix_counts[b]] = 1` (both single-window and multi-window paths) with a vectorised broadcast: `(torch.arange(K).unsqueeze(0) < gnn_prefix_counts.unsqueeze(1)).to(dtype)`. Eliminates the Python loop over batch dimension — zero output change, O(B×K) tensor op replaces O(B) Python iterations.

### A30 — `_word_embeddings` Fragile Hardcoded PEFT Path

**File:** `transformer_encoder.py`

Replaced the single hardcoded path `bert.base_model.model.embeddings.word_embeddings` with a property that tries three known PEFT internal paths in order (PEFT ≥0.4, some variants, older ≤0.3). Raises `AttributeError` with a PEFT version compatibility message if none yield an `nn.Embedding`. Path is validated at `__init__` time (after `get_peft_model`) — failure surfaces at construction, not at the first forward pass that activates prefix injection.

### A33 — `select_prefix_nodes` Python Loop Over Batch Dimension (Hybrid Vectorization)

**File:** `sentinel_model.py`

Pre-computed priority scores (`prio_scores`) and secondary sort scores (`sec_scores`) as tensors outside the per-graph loop using tensor operations over all N nodes at once. Inner loop now only does per-graph mask extraction and top-K selection (inherently variable-size, cannot vectorize without padding). Combined sort score = `prio * 1e3 + sec` (tensor sort, replaces Python `sort()` on tuple list).

### A34 — Secondary Sort Uses Post-GAT Embedding, Not Raw Feature

**File:** `sentinel_model.py` — `select_prefix_nodes`

Added `raw_node_features: torch.Tensor` parameter (`graphs.x`). Replaced `g_embs[local_idx, _EXT_CALL_DIM]` (dimension 10 of the 256-dim post-GAT embedding, which has no relation to `external_call_count`) with `raw_node_features[node_idx, _EXT_CALL_DIM]` (the actual log1p-normalised `external_call_count` from the input graph). Both callers (`forward()` and `compute_prefix_attention_mean()`) now pass `graphs.x`.

### A25b — `compute_prefix_attention_mean` Discards `node_counts`

**File:** `sentinel_model.py`

Removed the `isinstance(gnn_prefix, tuple): gnn_prefix, _ = gnn_prefix` workaround. Now properly unpacks `gnn_prefix, gnn_prefix_counts` and passes `gnn_prefix_counts` to `self.transformer()` so padded prefix positions are correctly masked when computing the diagnostic attention mean. Diagnostic-only fix — no training impact.

### NF-8 — Empty Batch Guard Returns Inconsistent Aux Dict Keys

**File:** `sentinel_model.py`

Added `"phase2": torch.zeros(B, self.num_classes, device=dev)` and `"jk_entropy": torch.tensor(0.0, device=dev)` to `aux_zeros` in the empty-batch guard path. Previously the empty-batch dict had only 3 keys (`gnn`, `transformer`, `fused`) while the normal path has 5 — any trainer code accessing `aux["phase2"]` or `aux["jk_entropy"]` on an empty-batch epoch would raise `KeyError`.

### Gate 3.1 — torch.compile Re-Validation

- Pending: 2-epoch smoke test with `torch.compile(model, dynamic=True)` — blocks Run 5 launch.

---

## 28. Run 5 Training Log Specification (2026-06-02)

**File:** `docs/pre-run-fixes/SENTINEL-Run5-Training-Log-Specification.md`
**Plan ref:** Phase 4.6 of `SENTINEL-Run5-Actionable Implementation Plan.md`

### Purpose

Exhaustive specification for all logging that must occur during Run 5 training. Every log item maps to a known bug, RC, or risk from the unified preflight proposal. Required for full observability, anomaly detection, and post-mortem capability. Added to the implementation plan as Phase 4.6 with a mandatory Gate 4.6 (logger startup verification) that blocks Run 5 launch.

### Specification Sections

| § | Section | Items | Key rationale |
|---|---------|-------|---------------|
| 1 | Data Integrity & Label Health | 9 | Non-negotiable given A20 (label=0 poisoning) and archive/data-integrity requirement |
| 2 | NaN/Inf & Gradient Health | 8 | Kill-run critical given A38 (NaN corrupts Adam state permanently) |
| 3 | Training Dynamics | 11 | Loss components separately, LR confirmation (NF-4 style mismatch detection) |
| 3B | AUC & Probability Quality | 16 | **Critical for agent module** — ML outputs probabilities not hard labels; AUC-PR and Brier Score directly measure agent-input quality |
| 4 | Model-Specific Logs | 10 | JK weight entropy, aux head norms, per-layer GNN output norms |
| 5 | Temperature Calibration | 7 | Must confirm T freshly computed, not reused from previous run |
| 6 | Resource & VRAM | 9 | RTX 3070 8GB ceiling; supports Gate 5.3 (max_nodes=2048 VRAM test) |
| 7 | Graph-Specific Logs | 8 | CEI label distribution, edge type distribution, def_map audit (NF-10) |
| 8 | Epoch-Level Summary | 37 fields | Structured JSON record per epoch — 37 fields covering all key indicators |
| 9 | Alert-Grade Anomalies | 3 tiers | KILL RUN (NaN), WARN+SKIP BATCH (poisoned labels), WARN (VRAM/JK/AUC-PR < 0.1/Brier > 0.4) |
| 10 | Log Format & Infrastructure | — | `StructuredLogger` skeleton, 3 output streams, sampling strategy, 14-question post-run checklist |

### Section 3B — Why AUC/Calibration Metrics Are Primary

The ML module outputs vulnerability **probabilities**, not hard classifications. The agent module consumes these as input signals and reasons over the probability gradient. This makes threshold-independent metrics (AUC-ROC, AUC-PR) and calibration metrics (Brier Score, ECE) more important than threshold-dependent F1. AUC-PR is especially critical for imbalanced vulnerability labels — a model that ignores rare classes gets high AUC-ROC but near-zero AUC-PR. New alert thresholds: `AUC-PR < 0.1` per label → WARN, `Brier Score > 0.4` per label → WARN, `F1 improves but AUC-ROC degrades` → WARN.

### Implementation Location

Phase 4.6 of the plan specifies implementing `StructuredLogger` in `ml/src/training/training_logger.py` (new file) and wiring it into `trainer.py`. The logger must be active before epoch 0 completes — Gate 4.6 verifies this.

---

## 30. Run 5 Pre-Flight — Phase 4 Training Loop Fixes (2026-06-02)

**Files:** `ml/src/training/trainer.py`, `ml/scripts/train.py`, `ml/src/training/training_logger.py` (new)
**Plan ref:** Phase 4 of `SENTINEL-Run5-Actionable Implementation Plan.md`

### 4.1 — A35: `_FocalFromLogits` Moved to Module Level

`_FocalFromLogits` was defined as a closure inside `train()`, capturing `_focal` via closure. Closures are not picklable — incompatible with DDP and multiprocessing. Moved to module level as `_FocalFromLogits(focal: FocalLoss)` with stored state. Instantiated as `_FocalFromLogits(FocalLoss(...))` inside `train()`.

### 4.2 — A36: `compute_pos_weight` No Longer Re-Reads CSV

`compute_pos_weight` previously called `pd.read_csv(label_csv)` on every invocation, doing redundant I/O when the CSV was already loaded by `DualPathDataset`. Signature changed to accept `train_dataset: DualPathDataset` directly. Labels extracted from `dataset._label_map` (already in RAM) filtered to `dataset.paired_hashes` (training split only). No CSV reads at call time.

### 4.3 — NF-4: `--gnn-layers` CLI Default Fixed to 8

`scripts/train.py` had `default=7` for `--gnn-layers`. `TrainConfig.gnn_layers` defaults to 8. Any training launched without an explicit `--gnn-layers` flag silently built a 7-layer GNN — a different architecture from what Run 5 requires. Fixed to `default=8`. Warning logged if `args.gnn_layers != 8`.

### 4.4 — NF-9: `AdamW(fused=True)` CPU Crash Fixed

`AdamW(fused=True)` is CUDA-only — crashes immediately on CPU with a RuntimeError. Replaced with `fused=(device == "cuda" or device.startswith("cuda:"))`. Required for smoke tests, CI, and any CPU-only debugging session.

### 4.5 — A37: Threshold Sweep Made Configurable

Per-epoch threshold sweep (`tune_thresholds=True` hardcoded) was running 19 threshold evals per class per epoch — expensive on long runs. Added `threshold_tune_interval: int = 10` to `TrainConfig` and `--threshold-tune-interval` to `scripts/train.py`. Sweep runs every N epochs and always at the final epoch; `_cached_tuned_thresholds` is reused between sweeps so downstream code always sees tuned thresholds even on non-sweep epochs.

### 4.6 — StructuredLogger Implemented (Phase 4.6, Gate 4.6)

**New file:** `ml/src/training/training_logger.py`

`StructuredLogger` and `TrainingAbortError` per Training Log Specification §10.2. Three JSONL streams created at init:
- `step_metrics.jsonl` — per-step granular metrics
- `epoch_summary.jsonl` — 37-field epoch summary (Spec §8)
- `alerts.jsonl` — all three alert tiers with timestamps

**Implemented:**
- `log_startup()` — data integrity hash (§1.8) + archive verification (§1.9) once at startup
- `check_batch()` — poisoned-label WARN_SKIP (§1.1/9.2.1), NaN/Inf input WARN_SKIP (§1.5/9.2.2)
- `check_loss()` — loss NaN/Inf KILL → `TrainingAbortError` (§2.1/9.1.1)
- `check_parameters()` — param NaN/Inf KILL (§2.2/9.1.2)
- `check_adam_state()` — `exp_avg`/`exp_avg_sq` NaN KILL (§2.6/9.1.3)
- `check_grad_norm()` — rolling 100-step history; spike WARN at >100× rolling mean (§2.4/2.8/9.3.2)
- `compute_auc_metrics()` — per-label AUC-ROC/PR, macro/micro averages, epoch-over-epoch deltas; AUC-PR < 0.1 WARN (§3B.1–3B.6,12–13/9.3.6b)
- `compute_brier()` — per-label + overall Brier Score; >0.4 WARN (§3B.7–8/9.3.6d)
- `compute_ece()` — ECE pooled across all labels (§3.9)
- `compute_prob_stats()` — min/max/mean/std/p5/p50/p95 per label (§3B.10)
- `check_f1_auc_divergence()` — WARN when F1 improves but AUC-ROC degrades >0.02 (§3B.15/9.3.6c)
- `check_aux_head()` — weight/bias norms; near-zero WARN (§4.3/4.4/9.3.3)
- `check_jk_entropy()` — Shannon entropy of JK weights; collapse WARN (§4.2/9.3.4)
- `check_vram()` — peak VRAM WARN > 7500 MB (§6.3/9.3.1)
- `build_epoch_summary()` — assembles all 37 Spec §8 fields
- `log_step()` / `log_epoch()` — writes to respective JSONL streams

**Wired into `trainer.py`:**
- `StructuredLogger` created immediately after `mlflow.log_params()`
- `log_startup()` called with dataset paths and archive dir before first epoch
- `evaluate()` now returns `_y_true` and `_y_probs` arrays for logger computations
- Per-epoch block: AUC, Brier, ECE, prob stats, JK entropy, aux head norms, VRAM computed; `log_epoch()` called; `check_vram()` called; `close()` at training end
- `TrainConfig.log_dir` field added (default: `ml/logs/<run_name>`)
- `--log-dir` added to `scripts/train.py`

**Gate 4.6:** Three JSONL files created at logger init. Data integrity hash and archive verification written before epoch 1. Both pending runtime verification at Run 5 launch.

## 31. Run 5 Pre-Flight — Phase 5 Training Interventions & CLI Hardening (2026-06-02)

**Files:** `ml/scripts/train.py`, `ml/src/training/trainer.py`, `ml/scripts/vram_gate_test.py` (new)
**Commit:** 4af3761
**Plan ref:** Phase 5 of `SENTINEL-Run5-Actionable Implementation Plan.md`

### 5.1 — Verified aux_phase2_loss_weight Propagation

Confirmed end-to-end chain: `TrainConfig.aux_phase2_loss_weight = 0.10` → call site `trainer.py:1567` passes `aux_phase2_loss_weight=config.aux_phase2_loss_weight` → `train_one_epoch()` parameter (default 0.0 in signature, always overridden). Phase 2 embeddings flow through `aux_head_phase2` → weighted BCE → summed into total loss at `trainer.py:664`. No stuck 0.0 default at runtime.

### 5.3 — Size-Stratified Timestamp F1 in evaluate()

`evaluate()` now collects per-sample node counts during the validation loop via `torch.bincount(graphs.batch)`. After evaluation, computes Timestamp F1 separately for three strata (EXP-L7 boundaries):
- `f1_Timestamp_small` — contracts with < 100 nodes
- `f1_Timestamp_medium` — contracts with 100–300 nodes
- `f1_Timestamp_large` — contracts with > 300 nodes
- `n_Timestamp_{stratum}` — sample count per stratum (for context)

Reports are logged to MLflow alongside standard per-class F1. Zero training code risk (evaluation path only). Option B (adversarial size regularizer) deferred to Run 6.

### 5.4 — --fusion-max-nodes CLI Arg + VRAM Gate Script

**`scripts/train.py`:** Added `--fusion-max-nodes` (default 1024; `dest="fusion_max_nodes"`) wired to `TrainConfig.fusion_max_nodes`. Raise to 2048 via CLI only after Gate 5.3 passes and Phase 7 re-extraction is complete. `--weighted-sampler` choices extended to include `"timestamp-size"` (was missing from argparse, causing argparse validation failure at CLI level).

**`ml/scripts/vram_gate_test.py` (new file):** Realistic worst-case VRAM gate for IMP-D1 (`max_nodes=2048`). Builds full SENTINEL model, generates synthetic batch at `nodes_per_graph=max_nodes` (worst case), runs complete training step: forward + backward + `optimizer.step` + AMP scaler + grad clip. Decision thresholds for RTX 3070 8 GB:
- PASS: peak < 7,500 MB → proceed with `max_nodes=2048`
- WARN: 7,500–8,000 MB → reduce `batch_size` to 8
- FAIL: > 8,000 MB → fall back to `max_nodes=1536`

Usage: `TRANSFORMERS_OFFLINE=1 PYTHONPATH=. python ml/scripts/vram_gate_test.py [--max-nodes N] [--batch-size B]`

**Gate 5.3:** Pending runtime execution before Phase 7.

### 5.5 — NF-5: Auxiliary Loss CLI Args

Three args added to `scripts/train.py`:
- `--aux-phase2-loss-weight` (default 0.10) — wired to `TrainConfig.aux_phase2_loss_weight`; enables CEI Phase 2 auxiliary loss for Run 5
- `--aux-cei-loss-weight` (default 0.0) — Phase 7 placeholder; inert until Gate 7.5 passes and `aux_cei_loss_weight` is wired into `trainer.py`
- `--jk-entropy-reg-lambda` (default 0.005) — already existed from Phase 4; documented here for completeness

## 32. Run 5 Pre-Flight — Phase 4.6 Training Log Spec Gap Fixes (2026-06-02)

**Files:** `ml/src/training/training_logger.py`, `ml/src/training/trainer.py`
**Commits:** 9d9fc79, b0b37c1
**Plan ref:** Phase 4.6 of `SENTINEL-Run5-Actionable Implementation Plan.md`

Audit of `SENTINEL-Run5-Training-Log-Specification.md` against the existing implementation revealed ~60% of specified checks and calls were either stub (`pass`) or never wired into the training loop. All gaps fixed.

### training_logger.py — New / Fixed

- **`_loss_history` + `_loss_spike_count`** added to `__init__` — rolling deque (maxlen=100) and per-epoch spike counter required for §2.7 spike detection.
- **`check_loss()` spike detection:** replaced `pass` stub with actual `>5× rolling_mean` logic. Appends to `_loss_history` on every finite loss; increments `_loss_spike_count` and emits WARN alert on spike.
- **`check_inputs(graphs_x, edge_index, step, epoch)`** (new, §1.3/1.7): validates `graphs.x` feature dim == 11 (WARN), checks `edge_index.min() >= 0` (WARN_SKIP → skip batch).
- **`reset_epoch_counters()`** (new): resets `_loss_spike_count`; must be called at the start of each epoch.
- **`log_calibration(temperatures, ece_pre, ece_post, epoch)`** (new, §5): writes temperature scaling result to `epoch_summary.jsonl`.
- **`log_graph_stats(edge_type_counts, cei_label_dist, epoch)`** (new, §7): writes graph-level dataset statistics to `epoch_summary.jsonl`.

### trainer.py — Wiring Completed

**`train_one_epoch` signature + return type:**
- Added `slog: StructuredLogger | None = None` and `epoch: int = 0` params.
- Return type changed from `tuple[float, int, float]` to `dict` with keys: `avg_loss`, `nan_batch_count`, `last_gnn_share`, `epoch_main_loss`, `epoch_aux_loss`, `epoch_ph2_loss`, `last_grad_norm_total`, `grad_norm_max_layer`, `loss_spike_count`, `grad_zero_count`.

**Epoch-level accumulators added:** `_epoch_main_sum`, `_epoch_aux_sum`, `_epoch_ph2_sum`, `_epoch_n`, `_last_grad_norm_total`, `_last_grad_max_layer`, `_last_grad_zero_count`.

**Per-batch (pre-forward):** `slog.check_batch()` + `slog.check_inputs()` → `continue` if either returns True.

**Per-batch (finite loss path):**
- `slog.check_loss()` called for spike detection.
- `_epoch_main_sum`, `_epoch_aux_sum`, `_epoch_ph2_sum` accumulated from true loss components.

**At `log_interval` (inside `should_log` block):**
- `compute_grad_stats(model)` called — produces true total grad norm, (layer_name, norm) max, zero-grad count.
- Stored in `_last_grad_norm_total`, `_last_grad_max_layer`, `_last_grad_zero_count`.
- `slog.check_grad_norm()`, `slog.check_vram()`, `slog.log_step()` all called with real metrics.

**Every 50 optimizer steps:** `slog.check_parameters()` + `slog.check_adam_state()` wired.

**NaN-rate >0.5%:** `slog.alert(KILL, …)` raises `TrainingAbortError`. Epoch loop catches it, calls `_slog.close()`, re-raises.

**Epoch loop (`train()`):**
- `_slog.reset_epoch_counters()` called at start of every epoch.
- `train_one_epoch` wrapped in `try/except TrainingAbortError`.
- `slog=_slog, epoch=epoch` passed at call site.
- Result dict unpacked: `train_loss`, `nan_batch_count`, `last_gnn_share` extracted; full dict stored as `_epoch_stats`.

**`build_epoch_summary()` — 9 wrong fields corrected:**
| Field | Was | Now |
|-------|-----|-----|
| `grad_norm_total` | `last_gnn_share` (ratio) | `_epoch_stats["last_grad_norm_total"]` (true norm) |
| `grad_norm_max_layer` | `("gnn", last_gnn_share)` | `_epoch_stats["grad_norm_max_layer"]` (true layer) |
| `main_loss` | `train_loss` (same as total) | `_epoch_stats["epoch_main_loss"]` |
| `aux_loss` | `0.0` (hardcoded) | `_epoch_stats["epoch_aux_loss"]` |
| `label_dist_val` | `{}` (empty) | computed from `_y_true` via `label_dist_from_tensor` |
| `prediction_entropy` | `0.0` (hardcoded) | binary entropy of `_y_probs`: `mean(-p·log p-(1-p)·log(1-p))` |
| `loss_spike_count` | `0` (hardcoded) | `_epoch_stats["loss_spike_count"]` |
| `grad_zero_count` | `0` (hardcoded) | `_epoch_stats["grad_zero_count"]` |
| `total_loss` | same as `main_loss` | `train_loss` (total including aux scaling) |

**Gate 4.6:** Still pending runtime verification at Run 5 launch — three JSONL files created, data hash and archive verification written, no KILL alerts in epoch 0.

---

## 33. v9 Findings Validation + Code Fixes C-1/C-3/H-2/M-3/M-6/NF-6 + Run 5 Kill (2026-06-02)

**Files:** `ml/src/preprocessing/graph_extractor.py`, `ml/src/training/trainer.py`, `ml/src/training/training_logger.py`, `ml/src/models/gnn_encoder.py`, `ml/scripts/train.py`
**Commit:** `e205b5e`

A set of code-audit findings from `docs/v9_findings.md` was validated against the current source and applied. Run 5 was killed at epoch 2 because the v9 training data lacked the graph-level fixes (C-1/H-2 require re-extraction; completing Run 5 on corrupt data would waste resources).

### Validation summary

| Finding | Verdict | Action |
|---------|---------|--------|
| C-1: per-statement CFG features always hardcoded | CONFIRMED | Fixed — requires v10 re-extraction |
| C-3: `_raw` BCE forward pass never backpropagated | CONFIRMED | Deleted — immediate |
| H-2: ReferenceVariable lvalues silently skipped in DEF_USE | CONFIRMED | Fixed — requires v10 re-extraction |
| M-3: `loss_for_log` used `accum_steps` not `_actual_window` | CONFIRMED | Fixed — immediate |
| M-6: ECE last bin used `< hi` excluding p=1.0 | CONFIRMED | Fixed — immediate |
| NF-6: Phase 2 ablation sub-masks applied to unablated `edge_index` | CONFIRMED | Fixed — immediate |
| EMITS edge bug (only 12 edges) | STALE/INVALID | BUG-H7 already fixed; actual count 216,699 |

### C-1 — Per-statement CFG features (`graph_extractor.py`)

Dims [2] (`uses_block_globals`), [7] (`return_ignored`), [8] (`call_target_typed`), [10] (`external_call_count`) were hardcoded to 0.0/1.0/0.0 at every CFG node regardless of the actual IR ops in that node. Four per-node helper functions added:

- `_node_uses_block_globals(node)`: walks IR, checks `SolidityVariableComposed` reads for `timestamp/number/difficulty/etc.` (fixes Timestamp dark spot)
- `_node_return_ignored(node)`: checks if any call's lvalue is used by later ops in the same node
- `_node_call_target_typed(node)`: returns 0.0 for `LowLevelCall` or `address`-typed `HighLevelCall` receiver
- `_node_external_call_count(node)`: counts `LowLevelCall + HighLevelCall + Transfer + Send` (fixes DoS dark spot — Transfer was previously invisible)

**Impact:** Real signal for Timestamp and DoS vulnerability classes; requires full v10 re-extraction.

### H-2 — ReferenceVariable DEF_USE edges (`graph_extractor.py`)

`_add_def_use_edges` silently skipped writes where the lvalue was a `ReferenceVariable` (Slither uses this for mapping writes like `balances[msg.sender]`). Added `_resolve_lval()` closure with `points_to` / `points_to_origin` traversal to resolve `ReferenceVariable` back to the underlying `StateVariable` or `LocalVariable`.

**Impact:** Missing DEF_USE edges for reentrancy state-write patterns (e.g. `balances[msg.sender] -= amount` before the external call). Requires full v10 re-extraction.

### C-3 — Deleted wasted BCE (`trainer.py`)

Line `_raw = aux_loss_fn(_aux_masked["phase2"], labels)` computed a full BCE forward pass whose result was never read or backpropagated. Deleted. Saves ~2–4 MB VRAM per micro-batch.

### M-3 — `loss_for_log` window denominator (`trainer.py`)

`loss_for_log = loss.item() * accum_steps` replaced with `loss_for_log = loss.item() * _actual_window`. The last partial gradient accumulation window has fewer micro-batches than `accum_steps`, so the old code produced a scaled-up loss value at epoch boundaries.

### M-6 — ECE last bin boundary (`training_logger.py`)

```python
mask = (flat_p >= lo) & (flat_p < hi)          # old — excludes p=1.0
mask = (flat_p >= lo) & (flat_p <= hi if is_last else flat_p < hi)  # new
```
Samples with predicted probability exactly 1.0 were dropped from calibration calculation.

### NF-6 — Phase 2 ablation bypass (`gnn_encoder.py`)

`cf_only_ei` and `icfg_only_ei` were computed by masking the full unablated `edge_index`/`edge_attr`, not the already-filtered `phase2_ei`/`phase2_ea`. This meant `--phase2-edge-types` ablation experiments had no effect on the CF/ICFG sub-masks within Phase 2 layers. Sub-masks now applied to `phase2_ei`/`phase2_ea`.

### Run 5 killed

Run 5 was at epoch 2 with v9 data (missing C-1 and H-2 fixes). No valid Run 5 checkpoint saved. v9 graphs archived to `ml/data/archive/graphs_v9_pre_run6/`. v10 re-extraction launched immediately.

---

## 34. v10 Re-Extraction Launch + Script Defaults Alignment (2026-06-02)

**Files:** `ml/scripts/reextract_graphs.py`, `ml/scripts/train.py`, `ml/scripts/create_cache.py`, `ml/scripts/tune_threshold.py`, `ml/scripts/retokenize_windowed.py`, `ml/scripts/interpretability/utils.py`, `ml/src/training/trainer.py`
**Commit:** `aaa4e93`

### v10 Re-Extraction

v9 graphs archived via `rsync --remove-source-files` to `ml/data/archive/graphs_v9_pre_run6/`. v10 re-extraction launched:

```
PYTHONPATH=. TRANSFORMERS_OFFLINE=1 python ml/scripts/reextract_graphs.py
```

v10 graphs extract to `ml/data/graphs/` with C-1 (per-statement CFG features) and H-2 (ReferenceVariable DEF_USE) fixes active. Tokens at `ml/data/tokens_windowed/` are unchanged and reused.

### Script defaults aligned to v10

All scripts updated from v9/deduped defaults to v10:

| Script | Old default | New default |
|--------|-------------|-------------|
| `train.py` `--splits-dir` | `v9_deduped` | `v10_deduped` |
| `train.py` `--cache-path` | `cached_dataset_v9.pkl` | `cached_dataset_v10.pkl` |
| `train.py` `--weighted-sampler` | `positive` | `timestamp-size` |
| `trainer.py` `TrainConfig.cache_path` | `cached_dataset_deduped.pkl` | `cached_dataset_v10.pkl` |
| `create_cache.py` `--label-csv` | `multilabel_index_deduped.csv` | `multilabel_index.csv` |
| `create_cache.py` `--output` | `cached_dataset_deduped.pkl` | `cached_dataset_v10.pkl` |
| `tune_threshold.py` `--label-csv` | `multilabel_index_deduped.csv` | `multilabel_index.csv` |
| `tune_threshold.py` `--splits-dir` | `splits/deduped` | `v10_deduped` |
| `retokenize_windowed.py` `DEFAULT_INPUT` | `multilabel_index_deduped.csv` | `multilabel_index.csv` |
| `interpretability/utils.py` `--cache` | `cached_dataset_v9.pkl` | `cached_dataset_v10.pkl` |
| `interpretability/utils.py` `--splits-dir` | `v9_deduped` | `v10_deduped` |

### Cache archived

`ml/data/cached_dataset_v9.pkl` (2.2 GB) moved to `ml/data/archive/cached_dataset_v9.pkl`. No active training cache exists until `create_cache.py` is run post-extraction.

### Post-extraction pipeline

After v10 extraction completes (~41,576 graphs):
```bash
python ml/scripts/build_multilabel_index.py        # rebuild multilabel_index.csv from v10 graphs
python ml/scripts/create_cache.py                  # produces cached_dataset_v10.pkl
python ml/scripts/create_splits.py --splits-dir ml/data/splits/v10_deduped
python ml/scripts/validate_graph_dataset.py --check-contains-edges --check-control-flow --check-block-globals
# Launch Run 6
```

---

## 35. Run 7 Architecture + ISSUE-1–4 Fixes

**Period:** 2026-06-03
**Commits:** `416d0e0`, `e2ad84e`, `139ebbc`

### What changed

**Architecture fixes (BUG-R7-1/2, IMP-R7-1/2/3):**
- `gnn_encoder.py`: `nn.Embedding(13, 16)` type embedding; `_GNN_IN_DIM=27`; Phase 2 conv heads 1→4
- `sentinel_model.py`: aux_phase2 + CFG eye pool over CFG_NODE types [8-12]; 4th CFG eye; classifier widened 3×128→4×128=512 input
- `trainer.py` + `train.py`: `aux_phase2_loss_weight` 0.10→0.20; `ARCHITECTURE="four_eye_v8"`; `MODEL_VERSION="v8.1"`

**ISSUE-1 through ISSUE-4 fixes:**
- ISSUE-1: `cfg_eye_proj` moved to GNN param group (LR×2.5) — was in `_other_params` at base LR
- ISSUE-2: `cfg_eye_proj` and `aux_phase2` added to torch.compile submodule list
- ISSUE-3: predictor passes `fusion_max_nodes` from checkpoint config
- ISSUE-4: predictor passes `gnn_phase2_edge_types` from checkpoint config

### Run 7 launch

`GCB-P1-Run7-v10-20260603` — 4-eye architecture, v10 data, `gnn_prefix_k=0`. Best ep39 F1=0.3074 (in progress at time of writing).

---

## 36. Fix #35 — Safe Resume: RNG State + Full Optimizer Restore

**Period:** 2026-06-04
**File:** `ml/src/training/trainer.py`

### Problem

Prior runs experienced 5–10 epoch F1 regression after stop/resume. Root cause: `resume_model_only=True` was the default, which discards the Adam optimizer's momentum and variance buffers on resume. The optimizer restarts "cold" — without any accumulated gradient history — causing noisy updates until momentum rebuilds (~5-10 epochs).

A secondary issue: RNG states (torch/CUDA/numpy/python random) were not saved, so resumed training saw a different batch ordering than uninterrupted training would have.

### Fix

**`trainer.py` — Fix #35:**
1. `resume_model_only` default changed `True → False` — full resume (optimizer + scheduler + patience + best_f1) is now the default when `--resume` is passed
2. Checkpoint now saves: `rng_state`, `cuda_rng_state`, `numpy_rng_state`, `python_rng_state`, `tuned_thresholds`
3. Full resume path restores all four RNG states and the per-class threshold cache
4. `import random` added to imports

### Remaining variance on resume

CUDA non-deterministic ops (cuDNN, SDPA) mean results are not bit-for-bit identical to uninterrupted training. Statistically equivalent within normal seed variance. True bit-perfect determinism would require `torch.use_deterministic_algorithms(True)` (~30% throughput penalty) and is not implemented.

### Backward compatibility

Old checkpoints (pre-Fix #35) lacking the new keys resume gracefully — each key is read with `.get()` and silently skipped if absent. `resume_model_only=True` can still be passed explicitly on the CLI to force model-only loading when intentionally fine-tuning from a different run's weights.

---

## 37. Pre-Run 9 Audit Findings

**Period:** 2026-06-05
**Owner:** Ali (verified by friend audit + manual test inference)
**Source:** [`project_run8_audit_findings.md`](../../home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run8_audit_findings.md) (full report)

### Context

Run 8 (`GCB-P1-Run8-v10-20260605`) was killed at ep29 step 100 (KeyboardInterrupt). Best checkpoint at **ep27, val F1=0.2814** (fixed) / 0.2764 (tuned). Test-set tuned F1=**0.2307** — a regression vs Run 7's 0.3423. Friend's audit + manual test inference surfaced **10 verified findings (A–J)** that explain the regression. Pure architecture iteration without fixing these will not beat Run 7.

### Verified findings (summary)

| # | Finding | Impact |
|---|---------|--------|
| **A** | `--relabel-timestamp` NEVER applied to v10 CSV | Timestamp class is 72.5% noise (only 27.5% of Timestamp=1 graphs fire feat[2]) |
| **B** | Test-set precision degenerate for 9/10 classes | DoS predicts positive for 76.8% of all test rows (4,789/6,237) |
| **C** | Manual inference: model fires near-constant ~0.30-0.45 baseline | Both Run 7 and Run 8 fire "everything" on safe contracts; no clean class |
| **D** | `CALL_ENTRY` (edge type 8) only iterates `node.internal_calls` | External calls get nothing; 0% of training graphs have an `EXTERNAL CALL_ENTRY` edge |
| **E** | `_compute_uses_block_globals` misses `now` keyword (Solidity 0.4.x alias) | 27.5% of Timestamp=1 contracts don't fire feat[2] — invisible to model |
| **F** | IntegerUO unlearnable: arithmetic IR ops collapse into `CFG_NODE_OTHER(12)` | Schema has no `arithmetic-op` node type; 87.9% pre-0.8 → no `unchecked{}` to learn from |
| **G** | Manual test contracts massively OOD by size | Median 20 nodes vs training 90 nodes; bottom 1st-7th percentile |
| **H** | Predictor `_format_result()` ignores per-class tuned thresholds | Hardcoded 0.55 tier; manual eval display only |
| **I** | 87.9% of BCCC dataset is pre-0.8 Solidity (0.4–0.7) | `in_unchecked` was rightly dropped in v7 (no signal) |
| **J** | Model fires near-constant 0.30-0.45 baseline on safe contracts | Run 7 L4 + Run 8 manual test confirm — flag for ALL 10 classes |

### Test contract problem

The 20 manual test contracts in `ml/scripts/test_contracts/` are **not** a reliable primary benchmark:
- **Size OOD:** median 20 nodes / 40 edges (bottom 1st-7th percentile of training distribution)
- **Era OOD:** 100% use `pragma solidity ^0.8.0` with explicit `unchecked{}` blocks
- Training is **87.9% pre-0.8 Solidity** (0.4–0.7) where overflow was implicit

The model rightly cannot learn the `unchecked{}` vulnerability pattern that the test contracts trigger. **Use `ml/data/smartbugs-curated/` (143 real contracts, 100% pre-0.8) as the primary OOD benchmark instead.**

### Decision flow

1. Apply data + schema fixes (§38 below) → re-extract → retrain (Run 9)
2. Pure architecture iteration is paused until #1 is done
3. SmartBugs Curated becomes the primary OOD benchmark; test_contracts/ is documented as OOD

---

## 38. Pre-Run 9 Fixes

**Period:** 2026-06-06
**Commits:** `eec9323`, `1df0f68`, `a80a148` (and Run 9 launch follow-ups)
**Source:** [`docs/pre-run9-fixes/`](pre-run9-fixes/) (8 fix proposals + PIPELINE + TODO + README)
**Progress:** 5/8 fixes complete (Fixes #1, #2, #3, #4, #8 applied); #5, #6, #7 pending (post-training eval)

### Status table

| # | Fix | Status | Commit | Files | Re-extract? |
|---|-----|--------|--------|-------|-------------|
| **#1** | Timestamp relabel (`parents[2]` → `parents[3]`) | **DONE** | `eec9323` | `ml/scripts/archive/dedup_multilabel_index.py:64` | No |
| **#2** | Block-globals extraction (catch `now` + library wrappers) | **DONE** | `eec9323` | `ml/src/preprocessing/graph_extractor.py:574-636, 690-742` | Yes (v9) |
| **#3** | External CALL_ENTRY edge for `HighLevelCall`/`LowLevelCall` | **DONE** | `eec9323` | `ml/src/preprocessing/graph_schema.py:217, 406`; `graph_extractor.py` | Yes (v9) |
| **#4** | IntegerUO schema gap (`CFG_NODE_ARITH=13` + `in_unchecked_block` feat[11]) | **DONE** | `eec9323` + `1df0f68` | `graph_schema.py:160-178, 386-403`; `graph_extractor.py:393-403` | Yes (v9) |
| **#5** | Re-derive labels from Slither detectors (10 classes) | PENDING | — | new `ml/scripts/derive_slither_labels.py` | Partial |
| **#6** | Fix predictor tier-threshold bug (hardcoded 0.55 vs per-class tuned) | PENDING | — | `ml/src/inference/predictor.py:150-151, 710-715` | No |
| **#7** | Add `manual_test_smartbugs.py` benchmark | PENDING | — | new `ml/scripts/manual_test_smartbugs.py` | No |
| **#8** | Document `complexity` complexity-proxy bias | **DONE** | `4c45f97` | `docs/interpretability/SENTINEL-Understanding-Run7.md` | No |

### Smoke verification (all pass)

- `roulette.sol` (pre-0.8, uses `now`): feat[2]=**4.0** (was 0.0 in v8) ✓
- `simple_dao.sol`: EXTERNAL_CALL=**1**, +1 CFG_NODE_ARITH ✓
- `03_integer_overflow.sol` (0.8.x unchecked): feat[11]=**8.0** ✓
- `12_safe_contract.sol` (0.8.x safe): feat[11]=**0.0**, EXTERNAL_CALL=1 ✓
- 100-graph v9 sample: 85% have EXTERNAL_CALL, 31% have CFG_NODE_ARITH, all feat_dim=12 ✓

### Schema bump trigger

The re-extract was triggered by 3 schema changes (Fix #2 + #3 + #4), all bumping `FEATURE_SCHEMA_VERSION` from `v8` to `v9`. All v8 checkpoints are now invalid; Run 9 starts from a cold start (not a Run 8 resume). See [§39](#39-v9-schema-upgrades) for the full schema diff.

### Cross-cutting concerns (all applied)

- **Fresh build:** all graphs, tokens, splits, cache rebuilt from scratch
- **Schema version bump:** `FEATURE_SCHEMA_VERSION = "v9"` in `ml/src/preprocessing/graph_schema.py:160`
- **DOC APPLIED banners:** 02-block-globals-extraction.md, 03-external-call-entry.md, 04-integeruo-schema-gap.md, README.md all updated in commit `4c45f97`
- **Run 9 gating:** see [§40](#40-run-9-launch--watcher)

### Run 9 gating criteria (post-fix)

1. Re-derive IntegerUO labels from Slither `integer-overflow` detector + structural guard (post-eval)
2. Re-validate Timestamp labels via `--relabel-timestamp` + `now`/library extraction fix
3. Re-validate `CALL_ENTRY`/`RETURN_TO` coverage: at least 5% of training graphs should now have an `EXTERNAL CALL_ENTRY` edge
4. SmartBugs Curated baseline: per-class precision > 0.3 on 6/10 classes (currently 1/10)
5. Manual safe contracts: no class above 0.50 (currently 5/10 classes fire > 0.50 on `12_safe_contract.sol`)

---

## 39. v9 Schema Upgrades

**Period:** 2026-06-06
**Commits:** `eec9323`, `1df0f68`
**Authoritative source:** [`docs/ml/adr/0001-schema-versioning.md`](ml/adr/0001-schema-versioning.md) (ADR-0001)
**Bumped:** `FEATURE_SCHEMA_VERSION = "v8"` → `"v9"` in `ml/src/preprocessing/graph_schema.py:160`

### Schema diff (v8 → v9)

| Constant | v8 | v9 | Reason |
|----------|----|----|--------|
| `FEATURE_SCHEMA_VERSION` | `"v8"` | `"v9"` | Marks the cache invalidation boundary |
| `NODE_FEATURE_DIM` | 11 | **12** | New feat[11] = `in_unchecked_block` (Slither 0.10 `node.scope.is_checked`) |
| `NUM_NODE_TYPES` | 13 | **14** | New `CFG_NODE_ARITH=13` for pure Binary arithmetic ops |
| `NUM_EDGE_TYPES` | 11 | **12** | New `EXTERNAL_CALL=11` self-loop on CFG nodes that make cross-contract calls |
| `_MAX_TYPE_ID` (sentinel_model.py) | 12.0 | **13.0** | Mirrors `max(NODE_TYPES.values())`; module-level assert enforces |
| `feat[2]` `uses_block_globals` | 8% fires | **9.1% fires** | Extended to catch `now` keyword, `block.timestamp`/`block.number`/`block.prevrandao`/`basefee`/`difficulty`/`blockhash`, library wrappers |

### Cache + dataset impact

- `ml/data/graphs/` — 41,576 .pt files re-extracted (v9 schema); v8 graphs moved to `ml/data/archive/graphs_v8_pre_run6/` for reference
- `ml/data/cached_dataset_v9.pkl` — 2.6 GB, all features in v9 shape
- All v8 checkpoints **invalid** for v9 (mismatched feature dim, edge type vocab, type embedding)
- Run 9 starts from a cold start (Run 8 checkpoint not used; see [§40](#40-run-9-launch--watcher))

### Slither API discoveries (avoid future schema-doc drift)

- `op.in_unchecked_block` does **NOT** exist on `Operation`. Use `node.scope.is_checked` (Slither 0.10, `slither/solc_parsing/declarations/function.py:1090`).
- `NodeType.STARTUNCHECKED` does **NOT** exist in installed Slither 0.10 — the v6 docstring was wrong.
- `BinaryType` member names are full words: `ADDITION`/`SUBTRACTION`/`MULTIPLICATION`/`DIVISION`/`MODULO`/`POWER`/`LEFT_SHIFT`/`RIGHT_SHIFT` (NOT the shorter `ADD`/`SUB`/`MUL` referenced in earlier docs).
- Empirically: pre-0.8 Solidity → ALL nodes have `is_checked=False` → feat[11]=1.0 universally. `feat[11]` becomes a **Solidity-era proxy** for the 87.9% pre-0.8 training subset, not a fault. (See [§38 finding I](#38-pre-run-9-fixes) — not a regression, was deliberately dropped in v7 BUG-L2.)

### Authoritative schema decisions

See [`docs/ml/adr/`](ml/adr/) for the *why* behind the schema versioning and class formulation:
- **ADR-0001** — Schema versioning (`FEATURE_SCHEMA_VERSION` string, cache key suffix, module-level asserts)
- **ADR-0002** — Multi-label formulation (10-class sigmoid multi-hot, `class_9` reserved)

---

## 40. Run 9 Launch + Watcher

**Period:** 2026-06-06 (in flight)
**Commits:** `a80a148` (Run 9 launch + watcher)
**Source scripts:** `ml/scripts/run9_launch.sh`, `ml/scripts/run9_watcher.sh`
**Log:** `/tmp/run9_v11.log` (also Windows toasts via watcher)

### Run configuration

| Field | Value |
|-------|-------|
| Run name | `GCB-P1-Run9-v11-20260606` |
| Schema | v9 (`FEATURE_SCHEMA_VERSION="v9"`) |
| Checkpoint (cold start) | None (Run 8 checkpoint invalid for v9) |
| Launched | 2026-06-06 ~03:30 UTC |
| Watcher PID | 2845414 |
| Speed | ~35 min/epoch on RTX 3070 8GB |

### Live status (as of 2026-06-06 ~14:00 UTC)

- **Epoch:** ~ep14 (in flight)
- **Best F1-macro:** 0.2476 at ep12
- **Loss:** 0.1265

**Top 3 classes (best F1):** IntegerUO=0.595, GasException=0.316, MishandledException=0.268
**Bottom 3 classes:** DenialOfService=0.015, Timestamp=0.145, UnusedReturn=0.201

### Gate criteria

Run 9 must beat both of:
- **Run 7 tuned F1-macro = 0.3423** (target)
- **Run 8 plateau F1-macro = 0.2307** (regression floor — must improve at minimum)

### Known unknowns

- **DoS:** bottom class with F1=0.015. The 98.6% co-occurrence with Reentrancy (BCCC label noise) means the model has only 14 DoS-only examples. Per-class F1 ceiling expected to stay at 0.15–0.30. See [ADR-0005](ml/adr/0005-bccc-dataset-choice.md) (Consequences: DoS/Reentrancy inseparability).
- **Timestamp:** relabeled (`--relabel-timestamp` applied) but feat[2] extension only helps pre-0.8 contracts; SmartBugs Curated (100% pre-0.8) is the OOD test.
- **SmartBugs Curated benchmark:** not yet measured (Fix #7 pending — `manual_test_smartbugs.py` script).

### Related

- [§38 Pre-Run 9 Fixes](#38-pre-run-9-fixes) — the 5 fixes that preceded this run
- [§39 v9 Schema Upgrades](#39-v9-schema-upgrades) — the schema Run 9 trains on
- [`docs/training/GCB-P1-Run7-analysis-2026-06-04.md`](training/GCB-P1-Run7-analysis-2026-06-04.md) — the analysis Run 9 must beat

---

## 41. Tier 1 Architectural Decision Records

**Period:** 2026-06-06
**Source:** [`docs/ml/adr/`](ml/adr/) (6 ADRs + INDEX + README + template)
**Format:** MADR-lite (~80–150 lines each), 4-digit zero-padded naming
**Status:** All 6 Tier 1 ADRs Accepted. Tier 2 (0007–0012) deferred to a future session.

### ADRs authored

| # | Title | One-line summary |
|---|-------|------------------|
| [ADR-0001](ml/adr/0001-schema-versioning.md) | Schema versioning | `FEATURE_SCHEMA_VERSION` string constant, cache key suffix, module-level asserts |
| [ADR-0002](ml/adr/0002-multi-label-formulation.md) | Multi-label formulation | 10-class sigmoid multi-hot; `class_9` reserved (SENTINEL phase 2, off by default) |
| [ADR-0003](ml/adr/0003-dual-path-four-eye-architecture.md) | Dual-path GNN+CodeBERT Four-Eye architecture | GNN + LoRA-CodeBERT cross-attend; 4 eyes (GNN/TF/Fused/CFG) summed for final logits |
| [ADR-0004](ml/adr/0004-three-phase-gat-routing.md) | Three-phase GAT routing | 8-layer GAT split Ph1/Ph2/Ph3; Ph2 sub-routing for `EXTERNAL_CALL`; JK attention aggregation |
| [ADR-0005](ml/adr/0005-bccc-dataset-choice.md) | BCCC-SCsVul-2024 as primary dataset | 41,576 deduped contracts; SmartBugs-curated held out as OOD benchmark; 87.9% pre-0.8 |
| [ADR-0006](ml/adr/0006-loss-formulation.md) | Loss formulation | ASL γ⁻=2 γ⁺=1 per-eye + aux BCE pathway (0→0.30 over 8ep) + 0.005 JK entropy |

### Tier structure

| Tier | Status | Scope |
|------|--------|-------|
| **Tier 1** | **Done** (this section) | Foundational design choices. Locks the architecture. |
| **Tier 2** | Deferred to future session | Implementation details, sub-system design |
| **Tier 3** | Backlog | Hyperparameters, training recipes, calibration thresholds |

### Tier 2 backlog (deferred)

- **0007:** Slither IR as the canonical extraction source (vs hand-rolled AST walk)
- **0008:** Windowed tokenization strategy (linspace subsample, stride 256, max 4 windows)
- **0009:** Cache architecture (`.pkl` vs `.parquet` vs LMDB, invalidation by schema version)
- **0010:** Pre-flight gate methodology (smoke tests 1–8, gate criteria for new runs)
- **0011:** Sampling strategy (WeightedRandomSampler, BCCC class imbalance mitigation)
- **0012:** Training kill criteria (F1-macro regression, aux BCE explosion, JK collapse)

### Code cross-links (one-line comments)

To make the ADRs discoverable from the source code, single-line `# See ADR-NNNN` comments were added in:
- `ml/src/preprocessing/graph_schema.py:158` → ADR-0001
- `ml/src/preprocessing/graph_extractor.py:3` → ADR-0001
- `ml/src/models/sentinel_model.py:145-146` → ADR-0003, ADR-0004
- `ml/src/training/losses.py:49` → ADR-0006

### Memory cross-link

`~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md:185` now references `docs/ml/adr/INDEX.md` as the authoritative source for the *why* behind SENTINEL's current architecture.

### Related

- [`docs/ml/adr/INDEX.md`](ml/adr/INDEX.md) — the ADR index (with status, date, one-line summary for each)
- [`docs/ml/adr/README.md`](ml/adr/README.md) — ADR lifecycle and tier structure
- [`docs/ml/adr/_template.md`](ml/adr/_template.md) — MADR-lite template for new ADRs

---

## 42. Run 9 v11 — Crash + Resume + Lambda Typo Incident (2026-06-06)

### Summary

Run 9 v11 reached its best result so far (F1-macro=**0.2586** at ep14) before crashing at ep16 step 300/455 when the VS Code terminal session was terminated at 15:49 UTC. A resume command was issued, but it contained a hyperparameter typo (`--jk-entropy-reg-lambda 0.0075` instead of the original `0.005`, a 50% increase). The new ep1 result (F1=0.2395) was saved as a "new best" and **overwrote the ep14 .pt file**. The metric 0.2586 survives in the structured log only; the weights are unrecoverable.

A second resume was issued with the correct lambda 0.005, starting from the F1=0.2395 checkpoint (preserved as `/tmp/run9_v11_ep1_2395.pt`). New training started at 16:46:23 UTC. ep1 validation expected at ~17:23 UTC.

### Timeline

| Time (UTC) | Event |
|---|---|
| 14:42 (≈) | ep14 best — F1=0.2586, checkpoint saved |
| 15:00 (≈) | ep15 — F1=0.2540, no improvement, patience=1/30 |
| 15:49 | **ep16 CRASHED** — VS Code terminal close → step 300/455 killed mid-epoch |
| 15:56 | First resume — `--jk-entropy-reg-lambda 0.0075` (TYPO, should be 0.005); `--resume-model-only` defaulted to True; best_f1 reset to 0.0; fresh optimizer/scheduler |
| 16:37 | ep1 result (typo'd lambda) — F1=0.2395; saved as new best; ep14 best .pt **GONE** |
| 16:42 | Damage assessment — Confirmed: trainer.py:1198-1200 + 2000 explain the overwrite; ep14 best (F1=0.2586) only in structured log |
| 16:46 | Second resume — Correct lambda 0.005; backup saved at `/tmp/run9_v11_ep1_2395.pt`; new training PID 3362523 |
| 17:23 (est) | ep1 result (correct lambda) — Awaiting |

### What Was Lost

| Asset | Before resume | After resume | Recoverable? |
|---|---|---|---|
| `GCB-P1-Run9-v11-20260606_best.pt` (F1=0.2586) | 281 MB, ep14 weights | OVERWRITTEN with ep1 (F1=0.2395, lambda 0.0075) | NO (in-memory only via structured log) |
| `GCB-P1-Run9-v11-20260606_best.state.json` | `{epoch:15, patience:1, best_f1:0.2586}` | `{epoch:1, patience:0, best_f1:0.2395}` | NO |
| Structured log | All historical metrics | Appended with new ep1 metrics | YES (F1=0.2586 still in log) |
| MLflow artifacts | None for Run 9 v11 | None for Run 9 v11 | N/A |
| Backup of F1=0.2395 (typo'd lambda) | N/A | `/tmp/run9_v11_ep1_2395.pt` (MD5 `26be38a4...`) | YES |

**Critical loss:** F1=0.2586 as a recoverable .pt file. The new run must re-discover this trajectory.

### Why the Overwrite Happened

Two compounding factors in `ml/src/training/trainer.py` and `ml/scripts/train.py`:

1. **`--resume-model-only` defaults to `True`** (`ml/scripts/train.py:255-258` — `default=True`). The user's resume command did not pass `--no-resume-model-only`, so the trainer took the model-only path. **It loads model weights, but initializes `best_f1 = 0.0` and `patience_counter = 0` from scratch** (`trainer.py:1198-1200`).
2. **The save condition uses the (reset) `best_f1`** (`trainer.py:2000`): `if val_metrics["f1_macro"] > best_f1:`. With `best_f1=0.0`, ANY positive F1 triggers a save. The new ep1 (F1=0.2395) was saved as a "new best", overwriting the loaded ep14 weights.

This is a **design choice** in the trainer. The `--resume-model-only` mode is intended for fine-tuning from a pretrained model — it explicitly does NOT carry over the old best_f1, because the new fine-tuning regime has its own metrics. But it is also the most common resume mode, and users may not realize the consequence.

### The Typo

The first resume command was issued with `--jk-entropy-reg-lambda 0.0075`. The original Run 9 launch config used `--jk-entropy-reg-lambda 0.005`. The 0.005 value was chosen because it balanced JK attention diversity without forcing unnatural uniformity (per Run 7/8 experience). A 50% increase has unknown effects on the attention distribution across phases 1/2/3.

**Root cause:** the resume command was typed by hand instead of being extracted from `ml/scripts/run9_launch.sh`. Cross-checking all hyperparameters line-by-line against the original launch config would have caught this.

### Mitigation Taken

1. ✅ **Backed up** ep1 with the typo'd lambda at `/tmp/run9_v11_ep1_2395.pt` (MD5 `26be38a4...`).
2. ✅ **Restarted training** with correct lambda 0.005 from the F1=0.2395 checkpoint (new training PID 3362523, started 16:46:23).
3. ✅ **Watcher** left running (PID 3094712) — tails `/tmp/run9_v11.log` for both runs.

### Future Safety Nets (deferred — see [project_run9_resume.md](../../../../home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run9_resume.md) for details)

1. **Hyperparameter diff in resume:** add a check in `trainer.py` that compares CLI args to the checkpoint's saved config and warns on mismatch. The checkpoint already saves `config` (see `trainer.py:2020-2030`).
2. **Default `--resume-model-only=False`:** change the default in `ml/scripts/train.py:255-258` to `False` so that full resume (with optimizer state, best_f1, patience) is the default. Users opt in to model-only resume.
3. **Best checkpoint versioning:** save checkpoints with epoch number in the filename (e.g., `GCB-P1-Run9-v11-20260606_ep14_F1-0.2586.pt`) and keep `best.pt` as a symlink. Old weights are never truly overwritten.
4. **Launch script as the resume source of truth:** write a `ml/scripts/resume_run9.sh` that reads the launch config and issues a resume command. Single source of truth.
5. **VS Code terminal close recovery:** add a `trap` to the training script that catches SIGTERM/SIGHUP and gracefully exits, saving the in-progress checkpoint. Currently killed mid-epoch with no save.

### Recovery Target

Ep1 with lambda 0.005 should land in 0.23-0.25 range. Ep2+ must climb back above 0.2586 to re-establish the best. The original trajectory took 14 epochs from random init to reach 0.2586. The new run starts from F1=0.2395 with fresh optim, so recovery is plausible within another 10-15 epochs (~6-9 hours).

### Related

- [`project_run9_resume.md`](../../../../home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run9_resume.md) — full incident narrative
- MEMORY.md "Current State" section — current Run 9 v11 status
- `ml/scripts/train.py:255-258` — `--resume-model-only` default
- `ml/src/training/trainer.py:1198-1200, 2000` — best_f1 init at training start + save condition
- `ml/logs/GCB-P1-Run9-v11-20260606.log` — append-only structured log (F1=0.2586 historical record)
- `/tmp/run9_v11_ep1_2395.pt` — backup of F1=0.2395 weights (typo'd lambda)
- §40 — Run 9 launch (where the lambda was originally set to 0.005)
- ADR-0006 — Loss formulation, including the 0.005 JK entropy value

---

## 43. BCCC-SCsVul-2024 Deep Dive — Phase 1 (2026-06-06)

### Summary

A structured deep-dive into the BCCC-SCsVul-2024 dataset, motivated by ADR-0005's reliance on it. Phase 1 (exploration) is complete. Phase 2 (validation + clean) is planned but not started. **No source files were modified; the dataset is treated as read-only.**

### Phase 1 Findings (Top 10)

1. **68,433 unique contracts, not 111,897.** 38.8% of files are exact byte-identical copies placed in multiple "candidate" folders.
2. **Multi-label, not single-class.** 41% of contracts have ≥2 simultaneous vulnerability labels. Matches SENTINEL's 12-binary-head design (ADR-0002).
3. **12 BCCC classes vs SENTINEL's 10.** `TransactionOrderDependence` (5.2%) and `WeakAccessMod` (2.8%) are NOT in SENTINEL. **D-F1 decision pending** (recommendation: drop 2 from training, keep 12 columns in `contracts_clean.csv` for future v2).
4. **766 contracts are `NonVulnerable` + have vulnerabilities.** Likely a meta-label semantic ("not audited") or label noise. Needs Phase 2 manual inspection.
5. **Top co-occurrence: `DenialOfService + Reentrancy` = 12,381 contracts (18% of corpus).** Massive head correlation; SENTINEL's ASL loss alone doesn't decouple this.
6. **CSV md5 verified** (`e38a2aa1c2b8a93c6cf8b23d2d7b870a`); per-file content integrity **NOT verifiable** (`Sourcecodes.md5` validates a missing ZIP). Trust assumption required.
7. **92% pre-0.6 Solidity** (mostly 0.4.x and 0.5.x). SENTINEL must compile these old solc versions; modern features (receive/fallback) won't exist.
8. **0% SPDX license headers** across 1,200-file sample. Pre-dates SPDX adoption.
9. **Multi-folder distribution:** 40,267 unique contents in 1 folder, 19,068 in 2, ..., 2 in 9 folders. Top contracts are famous templates (SafeMath, OpenZeppelin ERC20) flagged by 9 separate vulnerability detectors.
10. **CSV "ID" is a 64-hex hash but NOT sha256(file_content)** (95.5% mismatch). Likely keccak-256 of bytecode. Treated as opaque handle for dedup.

### Phase 1 Artifacts

| File | Lines | Size | Purpose |
|---|---:|---:|---|
| `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/01_exploration_inventory.md` | 435 | 27 KB | Full inventory: 8 sections, 10 findings, scorecard |
| `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/02_validation_deep_dive_plan.md` | 342 | 22 KB | Phase 2 plan: 8 workstreams, ~27 h, 9 risk register |
| `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/scripts/bccc_phase1_explore{1,2,3,4}.py` | 639 | 23 KB | 4 read-only reproducible scripts |

Total: 1,416 lines of artifacts.

### Phase 2 Plan (8 Workstreams)

| WS | Title | Est. (h) | Depends on |
|---|---|---:|---|
| A | Integrity & Dedup | 2.0 | — |
| B | Label Validation (paper lookup, manual inspections) | 8.0 | A |
| C | Compilation Probing (solc 0.4.x/0.5.x toolchain) | 3.25 | B |
| D | Cross-Corpus Overlap (BCCC vs SmartBugs-curated) | 2.5 | A |
| E | Per-Class Complexity Profile | 3.0 | A |
| F | Class Reconciliation (D-F1: 10 vs 12 classes) | 2.0 | B |
| G | Stratified Split Design (multi-label) | 2.5 | A, F |
| H | Final Cleaned Dataset (manifest, parquet, metadata) | 3.5 | A-G |
| **Total** | | **26.75 h** | **8-10 sessions** |

### Blocking Decision: D-F1

Before WS-G and WS-H can finish, user must approve one of:
- **(A)** Drop the 2 BCCC classes (TOD, WeakAccessMod) from SENTINEL training — keep SENTINEL's 10-class plan stable.
- **(B)** Add the 2 classes to SENTINEL — 12 binary heads, 12-fold output, ADR-0005 update.
- **(C)** Train on all 12, mask 2 at inference — hybrid; slight compute waste.

**Recommendation: (A)** — keeps SENTINEL v1 stable, can revisit in v2.

### Related

- [`Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/01_exploration_inventory.md`](../../Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/01_exploration_inventory.md) — Phase 1 full inventory
- [`Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/02_validation_deep_dive_plan.md`](../../Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/02_validation_deep_dive_plan.md) — Phase 2 plan
- MEMORY.md L186 — cross-link to deep dive
- ADR-0005 — BCCC-SCsVul-2024 dataset choice (relied on by this deep dive)
- `ml/data/BCCC-SCsVul-2024_README.md` — legacy weak-hint doc (claims "11 vulns" + 111,897 contracts — both misleading per Phase 1)

---

## 44. BCCC-SCsVul-2024 Deep Dive — Phase 2 — Final Cleaned Dataset (2026-06-06)

### Summary

Phase 2 of the BCCC-SCsVul-2024 deep dive is complete. All 8 workstreams (WS-A through WS-H) finished in a single ~2.5-hour session. Output: a production-ready cleaned dataset (`contracts_clean.csv` + `.parquet`, 67,311 contracts × 24 columns) that aligns BCCC's 12-class labels with SENTINEL's 10-class v9 schema, with 70/15/15 stratified splits and 766 review-pending contracts flagged for manual review.

This dataset is the **primary candidate to replace the v10 cached_dataset_v9.pkl** in future SENTINEL training (currently the Run 9 in-flight training uses the v10 cache, NOT this new dataset).

### Key Findings

1. **CSV is in LONG format** (not wide): 111,897 rows = 68,433 unique IDs × avg 1.635 classes/contract. Each row has exactly 1 positive class; the same ID appears multiple times with different single classes. After collapsing to one row per ID, the dataset is correctly multi-label.
2. **100% MATCH** between folder membership and CSV positive classes — the dataset is internally consistent.
3. **0 byte-identical overlap** with SmartBugs-curated (143 contracts) — confirms SmartBugs is clean OOD test data per ADR-0005.
4. **Compilation success rate: 73%** on a 100-contract stratified sample across 5 solc versions. Top error: PRAGMA mismatch (17/100), solvable by installing more solc versions.
5. **Class co-occurrence: heavy head correlation** — `DoS+Reentrancy` = 12,381 contracts (18% of corpus). ASL loss alone doesn't decouple this; consider a contrastive auxiliary loss in Run 10+.

### Decisions Applied

| Decision | Choice | Effect |
|---|---|---|
| **D-F1** | Drop WeakAccessMod (Class07) and TransactionOrderDependence (Class05) — no SENTINEL v9 equivalent | 1,122 contracts dropped; 67,311 kept |
| **D-B2** | Manual review of 766 NV+vuln contradictions | 766 flagged `review_pending=1`; held out from initial training |
| **D-D (auto)** | No overlap with SmartBugs-curated → use SmartBugs as OOD test set | Confirmed ADR-0005's OOD strategy |

### Workstreams Complete

| WS | Title | Status | Output |
|---|---|---|---|
| A | Integrity & Dedup | ✅ | `integrity/sha256_all_files.tsv` (16 MB), `dedup_map.csv` (68,433 rows) |
| B | Label Validation | ✅ | `labels/label_consistency.csv` (68,433), `class_cooccurrence.csv`, 3 sample files |
| C | Compilation Probing | ✅ | `compile/compile_results.csv` (100 rows), `compilation_report.md` (73% success) |
| D | Cross-Corpus Overlap | ✅ | `cross_corpus/overlap_report.md` (0 overlap) |
| E | Per-Class Complexity | ✅ | `complexity/per_contract_stats.csv` (68,433), `complexity_report.md` |
| F | Class Reconciliation | ✅ | `labels/contracts_filtered.csv` (67,311), `dropped_contracts.csv` (1,122), `review_pending_ids.csv` (766) |
| G | Stratified Split | ✅ | `splits/{train,val,test}.csv` (46,581 / 9,982 / 9,982) |
| H | Final Cleaned Dataset | ✅ | `outputs/contracts_clean.{csv,parquet}`, `split_assignments.csv`, `metadata.json`, `README.md` |

### Final Dataset Schema (24 columns)

- `id` — 64-hex keccak-256 of bytecode (BCCC's original ID)
- 10 class label columns (Class01..Class11 vulns + Class12 NV) — SENTINEL v9-aligned
- `primary_class` — first positive vuln class (single-label quick view)
- `n_pos` — number of positive classes (1-8)
- `is_pure_nv` — 1 if NV-only contract
- `review_pending` — 1 if D-B2 flagged (NV+vuln contradiction)
- `bccc_folder`, `bccc_file_path` — original source location
- `loc`, `n_functions`, `n_events`, `n_modifiers` — complexity stats
- `has_pragma`, `pragma`, `spdx` — header metadata

### Splits

| Split | n | % |
|---|---:|---:|
| Train | 46,581 | 70.0% |
| Val | 9,982 | 15.0% |
| Test | 9,982 | 15.0% |
| Held out (review_pending) | 766 | — |

Stratification: 2-stage on `(has_vuln, primary_vuln_class)`. **Approximation of iterative stratification** — pip install of `iterative-stratification` hung (network issue); when network is fixed, re-run WS-G for stricter stratification.

### File Hashes (for provenance)

| File | sha256 | Size |
|---|---|---:|
| `contracts_clean.csv` | `53b7b884c3ae38446bd3f1f0460c916d01e5a2b5ef96ee972eed7d8628f59e7a` | 17.3 MB |
| `contracts_clean.parquet` | `a60b43087d30f855c19864263c5d59978e2259920c40f8c9389d818f36630af6` | 9.2 MB |

### Bugs Discovered + Fixed Mid-Run

1. **WS-E:** `agg` variable shadowed `aggregate()` builtin → renamed to `agg_v`.
2. **WS-C:** `solc-select` not in subprocess PATH → hardcoded `ml/.venv/bin/solc-select` and `ml/.venv/bin/solc`.
3. **WS-C, WS-H:** pandas column-name mismatches (`Class01:ExternalBug` not `has_Class01:ExternalBug`; `loc_total` not `loc`).

### Phase 2 Artifacts (Total ~100 MB)

```
Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/
├── 00_session_log.md               (timeline + decisions log)
├── README.md                        (orientation)
├── scripts/                         (8 reproducible scripts, ~120 KB)
│   ├── a_integrity_dedup.py
│   ├── b_label_validation.py
│   ├── c_compile_probe.py
│   ├── d_cross_corpus.py
│   ├── e_complexity_profile.py
│   ├── f_class_reconciliation.py
│   ├── g_stratified_split.py
│   └── h_final_dataset.py
├── integrity/                       (31 MB)
├── labels/                          (21 MB)
├── complexity/                      (14 MB)
├── splits/                          (5.6 MB)
├── cross_corpus/                    (36 KB)
├── compile/                         (32 KB)
└── outputs/                         (30 MB; main deliverable)
    ├── contracts_clean.csv
    ├── contracts_clean.parquet
    ├── split_assignments.csv
    ├── metadata.json
    └── README.md
```

### Related

- [`Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/outputs/README.md`](../../Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/outputs/README.md) — final dataset usage guide
- [`Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/00_session_log.md`](../../Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase2_Validation_2026-06-06/00_session_log.md) — full timeline
- [§43](#43-bccc-scsvul-2024-deep-dive-phase-1) — Phase 1 inventory
- [§41 ADR-0005](#41-tier-1-architectural-decision-records) — BCCC dataset choice (relied on by this deep dive)
- MEMORY.md L186 — cross-link to deep dive
- MEMORY.md L187 — key findings (updated)
- [§42](#42-run-9-v11--crash--resume--lambda-typo) — Run 9 v11 incident (parallel work, not blocking Phase 2)

---

## 45. Model Evaluation Dashboard — v2 Spec Rewrite (2026-06-06)

### Summary

The `.kiro/specs/model-evaluation-dashboard/` spec (v1, dated 2026-06-04) was 60-70% aligned with current SENTINEL state. Per the alignment audit (`gap_analysis_2026-06-06.md`, 257 lines), 14 gaps and 9 improvements were identified. **A full v2 rewrite was completed in one session**, producing three new spec documents that supersede v1.

**Stack:** Python 3.12, FastAPI, Pydantic v2, sklearn, hypothesis (PBT). **New module:** `ml/src/api/` (separate from existing `ml/src/inference/api.py` 402-line production endpoint).

### Artifacts

| File | Lines | Contents |
|---|---:|---|
| `requirements.md` v2 | 532 | 39 requirements (21 updated from v1 + 18 new) |
| `design.md` v2 | 1185 | 10 components (6 updated from v1 + 4 new); 27 properties (15 original + 12 new) |
| `tasks.md` v2 | ~430 | 7 phases; ~60 tasks; 1 task-to-requirement traceability matrix |
| `gap_analysis_2026-06-06.md` | 257 | Audit (now marked DEPRECATED; all findings incorporated into v2) |
| `.config.kiro` | 1 | Updated to v2 with `version`, `supersedes`, `gapAnalysisRef` keys |

### 14 Gaps Addressed (from gap_analysis_2026-06-06.md)

| # | Gap | Resolution in v2 |
|---|---|---|
| 1 | Schema version mismatch (v8.1 vs v9) | Req 2 — hard requirement `EXPECTED_SCHEMA_VERSION = "v9"`; refuse non-v9 unless `--allow-legacy-schema` |
| 2 | `ml/src/api/` vs existing `ml/src/inference/api.py` | Decision: separate module (Phase 0.1); production = contract→verdict, eval dashboard = checkpoint→metrics |
| 3 | No schema version validation in checkpoint loading | Property 2 — refuses non-v9; warn + proceed if no version |
| 5 | Default threshold 0.50 is wrong | Req 1.6 — load `*_thresholds.json` sidecar if present; default 0.50 only if absent |
| 6 | Missing drift detection integration | Req 27 + new `DriftIntegrator` component (KS p<0.01 OR PSI>0.25) |
| 7 | Missing 3-tier output (CONFIRMED/SUSPICIOUS/NOTEWORTHY) | Req 26 + `compute_tier_counts()` in MetricsEngine |
| 8 | BCCC D-F1/D-B2 decisions not reflected | Req 24 — `review_pending=1` filter (excluded by default, `--include-review-pending` flag) |
| 9 | Missing SmartBugs-curated OOD benchmark | Req 25 + `evaluate_smartbugs()`; OOD excluded from cache |
| 10 | Missing ZK-circuit integration | Req 28 + new `ZKProvenanceProvider` (EZKL opt-in, 501 if not installed) |
| 11 | No data source guidance | Req 29 + new `DataSourceRegistry` (4 sources: bccc_cleaned_v1, legacy_v10, smartbugs_curated, custom) |
| 12 | `tuned_thresholds` is in sidecar (not embedded) | Req 1.6, 6.7 — load from `{stem}_thresholds.json` sidecar; pass to ThresholdTuner |
| 13 | test_contracts (ml/scripts/test_contracts/) not flagged as legacy | Req 30 — refuse unless `--allow-legacy-test-contracts` |
| 14 | Run 9 v11 incident not reflected | Req 23 — full `DataLineage` (train_run, epoch, commit_sha, seed, schema_version) for reproducibility |

### 9 Improvements Incorporated

| # | Improvement | v2 Component |
|---|---|---|
| 1 | Run Comparison view (Run 7 vs Run 8 vs Run 9) | Req 31 + `CheckpointManager.compare_runs()` (2-5 paths, per-class deltas + likely-cause hints) |
| 2 | BCCC source code context in error analysis | Req 32 + `ErrorAnalyzer.get_source_context()` (via `bccc_file_path` in `contracts_clean.csv`) |
| 3 | Complexity stratification (LOC, n_functions) | Req 33 + new `ComplexityStratifier` helper (Q1-Q4 per-class F1, flag bias if delta>0.10) |
| 4 | Multi-label-specific metrics | Req 34 + new `MultiLabelMetrics` helper (subset_accuracy, LRAP, coverage error — match sklearn) |
| 5 | Per-eye logging hooks | Req 35 + Evaluator saves GNN/TF/Fused/CFG/Main logits (≤5x cache overhead) |
| 6 | Single fix-priority output | Req 36 + new `DiagnosticsAggregator` (ranked by estimated F1 impact; static mapping 0.01-0.05) |
| 7 | Data lineage tracking | Req 23 + new `DataLineageTracker` (auto-detect commit_sha via `git rev-parse HEAD`) |
| 8 | CLI alternative | Req 37 + Phase 6 — `python -m ml.src.api.cli {eval,metrics,errors,diagnostics,...}` |
| 9 | Performance tracing | Req 38 + loguru INFO timings per phase; `performance` field in EvaluationResult |

### 10 Components (6 updated from v1 + 4 NEW in v2)

| # | Component | Status | New in v2? |
|---|---|---|---|
| 1 | CheckpointManager | Updated | (now incl. `compare_runs`, schema validation, threshold loading) |
| 2 | Evaluator | Updated | (now incl. data_source, review_pending filter, per-eye logging, ZK) |
| 3 | MetricsEngine | Updated | (now incl. multi-label, 3-tier, complexity stratification, edge ablation 12 v9) |
| 4 | ErrorAnalyzer | Updated | (now incl. 3-tier labels, BCCC `bccc_file_path` source lookup) |
| 5 | ThresholdTuner | Updated | (now starts from checkpoint sidecar) |
| 6 | CacheManager | Updated | (now data_source in cache key, OOD excluded) |
| 7 | **DataLineageTracker** | NEW | Full v2 |
| 8 | **DriftIntegrator** | NEW | Full v2 |
| 9 | **ZKProvenanceProvider** | NEW | Full v2 |
| 10 | **DiagnosticsAggregator** | NEW | Full v2 |
| Helper A | **MultiLabelMetrics** | NEW helper | subset_accuracy, LRAP, coverage error |
| Helper B | **ComplexityStratifier** | NEW helper | LOC + n_functions quartiles |
| Helper C | **DataSourceRegistry** | NEW helper | 4 sources + BCCC SHA-256 validation |

### 7 Phases / ~60 Tasks

| Phase | Title | Tasks |
|---|---|---:|
| 0 | Decisions | 3 |
| 1 | Schema upgrade + project skeleton | 5 |
| 2 | Core 6 components | 14 |
| 3 | New 4 v2 components | 5 |
| 4 | Data integration (BCCC/SmartBugs/multi-label/complexity) | 8 |
| 5 | Tests (27 properties + unit + integration) | 9 |
| 6 | CLI alternative | 4 |
| 7 | Documentation + release | 5 |
| **Total** | | **~60** |

### Key Module Placement Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Module location | `ml/src/api/` (NEW) | Separate from `ml/src/inference/api.py` (402L, production contract→verdict endpoint, 3-tier output, low latency) |
| FastAPI instance | New instance in `ml/src/api/main.py` | Different audiences, different SLOs, different deployment cycles |
| CLI tool name | `python -m ml.src.api.cli` | Reuses Manager classes; no duplicate logic |
| ZK provenance policy | Opt-in via `--zk-prove` | EZKL not installed by default; clear 501 if requested without EZKL |
| BCCC data source default | `bccc_cleaned_v1` | Newest, SENTINEL v9 schema-aligned (D-F1 applied) |
| SmartBugs OOD | `data_source=smartbugs_curated` | 143 contracts, 0 byte-overlap with BCCC; `exclude_cache=True` |

### Out of Scope for v2 (Deferred to v3+)

- Live training monitoring (training has its own logger; eval dashboard reads outputs post-hoc)
- ZK-prove every eval by default (kept opt-in)
- Multi-GPU inference (single-GPU sufficient for 67K contracts in research mode)
- Auto-scaling of `fusion_max_nodes` (flag for v3)

### Related

- [`.kiro/specs/model-evaluation-dashboard/requirements.md`](../../.kiro/specs/model-evaluation-dashboard/requirements.md) v2 (532L, 39 reqs)
- [`.kiro/specs/model-evaluation-dashboard/design.md`](../../.kiro/specs/model-evaluation-dashboard/design.md) v2 (1185L, 10 components, 27 properties)
- [`.kiro/specs/model-evaluation-dashboard/tasks.md`](../../.kiro/specs/model-evaluation-dashboard/tasks.md) v2 (~430L, 7 phases, ~60 tasks)
- [`.kiro/specs/model-evaluation-dashboard/gap_analysis_2026-06-06.md`](../../.kiro/specs/model-evaluation-dashboard/gap_analysis_2026-06-06.md) — original audit (now DEPRECATED banner)
- [.config.kiro](../../.kiro/specs/model-evaluation-dashboard/.config.kiro) — updated with v2 metadata
- MEMORY.md L189 (new) — v2 spec status
- [§44](#44-bccc-scsvul-2024-deep-dive-phase-2--final-cleaned-dataset) — BCCC Phase 2 (this spec references its `contracts_clean.csv` as the new `bccc_cleaned_v1` data source)
- [ADR-0001](#41-tier-1-architectural-decision-records) — Schema versioning (the spec implements this)

---

## §46 — BCCC Deep Dive Phase 5: Label Verification

**Date:** 2026-06-08
**Goal:** Verify ALL 67,311 BCCC labels using multi-method, gated approach before training SENTINEL.

### Background

Phase 4 Stage 1 revealed catastrophic label noise:
- Reentrancy: 89.4% false positives (BCCC flagged any external call + state change)
- CallToUnknown: 91% had no external calls at all
- ExternalBug: 100% FP in manual sample
- 3-way tool agreement F1 = 0.000

Training on these labels would teach the model to detect label noise, not vulnerabilities.

### Method (6 Stages)

| Stage | What | Gate |
|-------|------|------|
| 5.0 | Ground truth definitions for all 9 classes | — |
| 5.1 | Evidence integration (67,311 × 58 evidence table) | 3 clean classes verified manually (IntegerUO, UnusedReturn, MishandledException) |
| 5.2 | Automated verification on 6 noisy classes | Disputes identified per class |
| 5.3 | Discrepancy resolution (structural rules + manual review) | Residual CSVs produced |
| 5.4 | Manual extrapolation + per-contract verdicts | Gate results per class |
| 5.5 | GraphCodeBERT embedding + HDBSCAN propagation | **DEFERRED** — Run 9 GPU blocked |
| 5.6 | Synthesis → `contracts_clean_v1.3.csv` | ✅ Complete |

### Per-Class Results

| Class | Before | After | Retained | Gate | Method |
|-------|--------|-------|----------|------|--------|
| Reentrancy | 17,698 | 1,699 | 9.6% | VERIFIED ✅ | Regex `.call.value()` (99.8% high-conf) |
| CallToUnknown | 11,131 | 239 | 2.1% | PROVISIONAL → 5.5 | Regex `.call()` (87.9% high-conf) |
| Timestamp | 2,674 | 1,075 | 40.2% | BEST-EFFORT | Regex `block.timestamp` (52.6% conf) |
| ExternalBug | 3,604 | 344 | 9.5% | PROVISIONAL → 5.5 | Regex `selfdestruct`/`tx.origin` (93.1% conf) |
| GasException | 6,879 | 2,794 | 40.6% | PROVISIONAL → 5.5 | Slither `costly-loop` (80.8% conf) |
| DenialOfService | 12,394 | 1,252 | 10.1% | BEST-EFFORT | Slither `calls-loop` (64.5% conf) |
| IntegerUO | 16,740 | 16,740 | 100% | VERIFIED (clean) | Manual review 39/39 |
| UnusedReturn | 3,229 | 3,229 | 100% | VERIFIED (clean) | Manual review 10/10 |
| MishandledException | 5,154 | 5,154 | 100% | VERIFIED (clean) | Manual review 20/20 |
| NonVulnerable | 26,148 | 44,899 | +18,751 | — | Reclassified from noisy classes |

### Key Findings

1. **BCCC folder assignments were near-random for CallToUnknown** — 86.9% of 11,131 contracts had NO low-level call at all
2. **Reentrancy definition was too broad** — only `.call.value()` (10.6% of BCCC's Reentrancy) is true reentrancy
3. **Tool "LOW confidence" on clean classes = tool recall gap**, not label noise
4. **Timestamp and DoS have structural ambiguity** — best-effort, would benefit from Stage 5.5

### Outputs

- `contracts_clean_v1.3.csv` — 67,311 × 36 cols with verified labels
- `contracts_clean_v1.4.csv` — newer version with gap fixes
- `p5_s6_verification_report.md` — per-class gate results
- `p5_s6_class_size_comparison.csv` — before/after counts
- `review_batches/` — ~40 contracts per class for manual QA

### Impact on Training

**Run 9** (in flight) uses OLD noisy labels → will be "before" baseline.
**Run 10** (planned) will use `contracts_clean_v1.3.csv` → should show F1 improvement, especially on Reentrancy.

### Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D-P5-1 | Clean classes verified at Stage 5.1 (manual path) | Tool agreement F1=0.000 proves tools unreliable; manual review is ground truth |
| D-P5-2 | Timestamp reclassified as "moderate noisy" | 50% FP in sample, NOT clean as originally thought |
| D-P5-3 | ExternalBug in "Hard Noisy" only | 100% FP in sample, definition ambiguous |
| D-P5-4 | Per-class verification, not dataset-wide | Different classes verified at different stages |
| D-P5-5 | Stage 5.5 deferred | Run 9 GPU blocked; v1.3 usable without it |

### Related

- `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/` — all Phase 5 files
- `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/05_phase5_plan.md` — plan (612L)
- `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md` — handover doc
- MEMORY.md (Phase 5 section)


