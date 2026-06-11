# Stage 5 — Splitting + Registry

**Date:** 2026-07-14
**Status:** NOT STARTED. Reading required before Stage 6.
**Reading time:** 20-30 minutes.
**Goal:** After this doc, you can answer all 6 items in `LEARNING_CHECKLIST.md` §"Stage 5" from memory.

---

## 1️⃣ The Problem

### What Stage 5 has to deliver

Stages 1–4 produced verified labeled data. Stage 5 prepares it for **training**: splitting into train/val/test, and registering every artifact in a catalog so "what dataset version did Run 11 train on?" is answerable 6 months from now.

The BCCC failure had a splitting component too: 38.8% duplication meant many contracts appeared in both train and test, inflating Run 9's F1 by ~0.05. Stage 5 prevents this with a two-pass split + leakage auditor.

### The NonVulnerable 3:1 cap (friend review)

DISL has 514,506 unlabeled contracts. With ~1,200 positives from the 5 critical-path sources, the default ratio is 514K:1,200 = 428:1. This is the **same BCCC failure pattern at larger scale** — a model that defaults to "predict negative" and is right 99%+ of the time never learns positive patterns.

The cap is `pipeline.negative.positive_ratio_max: 3.0`. NonVulnerable is subsampled to at most 3× total positive count, stratified by source.

---

## 2️⃣ The Solution

### Four splitter strategies (D-5.1)

| Strategy | When to use | How it works |
|---|---|---|
| **Random** | Default for unlabeled sources | Random assignment to train/val/test |
| **Stratified** | Preserves per-class distribution | Each class is split proportionally |
| **Project-level** | Audit datasets (Bastet, ScaBench, Web3Bugs) | Entire project in one split |
| **Temporal** | Time-sensitive data | Split by date |

For audit datasets, **project-level** is the default (a project is entirely in one split). For tool-derived datasets, **stratified with project-level fallback**.

### Two-pass splitting (D-5.2)

Pass 1: stratified splitter assigns contracts. Pass 2: `dedup_enforcer` reassigns any near-dup group that straddles a split boundary. This prevents the BCCC 38.8% duplication leaking across train/test.

### Leakage auditor (D-5.3)

Independent post-split check using text similarity (different algorithm from AST similarity in dedup). Reports any leak the enforcer missed. The auditor is a safety net, not a block.

### SQLite + YAML catalog (D-5.4)

The catalog has 4 base tables + 2 system tables:
- `sources` — per-source pin + last-fetched
- `artifacts` — per-exported-artifact hash + lineage
- `splits` — per-split-version seed + strategy
- `dataset_versions` — named composite (source set + preprocessing config + split version)
- `schema_migrations` — tracks every schema change
- `dataset_version_retirements` — old versions marked "superseded" (audit trail is permanent)

Every DB write produces a corresponding YAML entry. CI checks they stay in sync.

### Hash verification (D-5.6)

The ML module's `SentinelDataset.__init__` calls `verify_artifact_hash()` before loading. If the hash doesn't match the registered hash, the load fails. This prevents "I edited the export file by hand and the model trained on the wrong data."

### Dataset versions are append-only (D-5.7)

A dataset version is a named promise: `sentinel-v2-gold-2026-08`. Once registered, immutable. Updates create new versions. The `dataset_diff` tool shows what changed between versions.

---

## 3️⃣ The Broader Context

### What Stage 5 enables downstream

- **Stage 6 (analysis)** reads splits for per-class distribution analysis
- **Stage 7 (export)** produces sharded export from the splits
- **Stage 8 (Run 11)** trains on the registered dataset version

### What breaks if Stage 5 is wrong

- Cross-split leakage → inflated F1 (same as BCCC 38.8% duplication)
- Missing NonVulnerable cap → 514K:1 imbalance → model defaults to "predict negative"
- Missing catalog → "what dataset version did Run 11 train on?" is unanswerable
- Missing hash verification → hand-edited export silently poisons training

---

## 4️⃣ Verification — Stage 5 exit criteria

| # | Check | Status |
|---|---|---|
| 1 | 4 splitter strategies produce correct splits | ⏳ |
| 2 | `dedup_enforcer` reassigns straddling groups | ⏳ |
| 3 | `leakage_auditor` reports 0 leaks on clean split | ⏳ |
| 4 | SQLite catalog with 4+2 tables | ⏳ |
| 5 | `load_artifact("sentinel-v2-dryrun-2026-08")` works | ⏳ |
| 6 | `verify_artifact_hash()` catches tampered files | ⏳ |
| 7 | NonVulnerable 3:1 cap enforced | ⏳ |

---

## 5️⃣ The "got it" checklist

1. **Why two-pass splitting?** Pass 1: stratified assignment. Pass 2: dedup_enforcer reassigns straddling near-dup groups. Prevents BCCC-style train/test leakage.

2. **What's the NonVulnerable 3:1 cap?** DISL's 514K contracts would create 428:1 imbalance. Cap at 3× total positives, stratified by source.

3. **Why project-level splitting for audit datasets?** A project (e.g. ScaBench's 31 projects) is entirely in one split. Prevents "90% of one project's contracts in train, 10% in test" bias.

4. **What's in the SQLite catalog?** 4 base tables (sources, artifacts, splits, dataset_versions) + 2 system tables (schema_migrations, dataset_version_retirements). YAML mirror for version control.

5. **Why hash verification at load time?** Prevents hand-edited exports from silently poisoning training. The hash is computed once at `__init__`, not per `__getitem__`.

6. **Why dataset versions are append-only?** The audit trail is permanent. The v1.4 BCCC labels, v8 BCCC graphs, v9 graphs — all preserved with `superseded_by` chain.

If you can answer all 6, Stage 5 is mastered.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 5"
- **06_stage_5_splitting_registry.md** — the design + intent document
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §3.6 (splitting), §3.7 (registry)

When you're ready, say **"Stage 5 is mastered — let's move to Stage 6."**
