# Data Module Proposals

**Date:** 2026-06-08
**Scope:** Sentinel v2 data module — proposal + 9 actionable plans for execution

This folder is the **single source of truth** for the Sentinel v2 data engineering module. If you want to know what the v2 build is, how it's structured, when it runs, and what to do next — start here.

---

## How to read this folder

```
docs/proposal/Data_Module_Proposals/
├── README.md                                         ← this file
├── Sentinel_v2_Data_Module_Integration_Proposal.md   ← BINDING proposal (the contract)
├── datasources_suggestions.md                        ← friend's research (applied into the binding proposal §6)
├── actionable_plans/                                 ← 9 executable plans (one per build stage)
│   ├── 00_INDEX.md                                   ← master plan linking everything
│   ├── 01_stage_0_skeleton.md
│   ├── 02_stage_1_ingest_preprocess.md
│   ├── 03_stage_2_representation.md
│   ├── 04_stage_3_labeling.md
│   ├── 05_stage_4_verification.md
│   ├── 06_stage_5_splitting_registry.md
│   ├── 07_stage_6_analysis.md
│   ├── 08_stage_7_export_seam.md
│   └── 09_stage_8_run11_launch.md
└── archive/                                          ← v1 drafts, applied patches (reference only)
    ├── AUDIT_PATCHES_applied_2026-06-08.md
    ├── first_proposal_data_module_v1_2026-06-08.md
    └── Sentinel_v2_Dataset_Proposal_v1_2026-06-08.md
```

**Read order:**
1. **`Sentinel_v2_Data_Module_Integration_Proposal.md`** (the binding proposal) — the *what* and *why*
2. **`actionable_plans/00_INDEX.md`** (master plan) — the *how* and *when*
3. **Stage plans 01–09** — the *detail*, one per build stage
4. **`datasources_suggestions.md`** (friend's research) — context for the 5 new data sources (DIVE, FORGE, SolidiFI, ScrawlD, Code4rena, DeFi Hacks REKT) that were added to the binding proposal §6
5. **`archive/`** — historical drafts and applied patches; **do not consult for current design**

---

## Status

| Item | Status |
|---|---|
| Binding proposal | ✅ v1.2 (2026-06-09) — 5 critical-path + 12 additive (v2.1) sources, post-friend-review; 10-week build |
| Audit patches | ✅ All applied; archived for reference |
| Stage 0 (skeleton) | ⏳ Pending — start 2026-06-09 |
| Stage 1–7 (build) | ⏳ Pending — Jun 16 – Aug 17 |
| Stage 8 (Run 11 launch) | ⏳ Pending — 2026-08-18 |
| Code | ⏳ No code yet; plans are design + intent only |
| Memory cross-link | ✅ Added to `~/.claude/projects/.../memory/MEMORY.md` "Memory Index" |

---

## The 30-second version

**Why this folder exists:** Run 9 (the latest training run) hit a ceiling at F1=0.31 because the BCCC dataset has 89% Reentrancy false positives and 86.9% CallToUnknown false positives. The model learned a "complexity proxy" instead of vulnerability patterns. The BCCC folder-based labeling is the root cause; the model architecture is not the bottleneck.

**What's being built:** a new `sentinel-data` package at `~/projects/sentinel/Data/` (which already contains the BCCC deep-dive outputs from Phases 1-5) that owns the entire data pipeline from raw contract ingestion through verified, versioned, multi-source, multi-label dataset export. The ML module (`sentinel-ml`) consumes the registry-versioned artifacts; it never sees a raw contract.

**The 9 submodules** (per the binding proposal §2, **revised 2026-06-09 post-friend-review**):
- **Ingestion** — per-source connectors (Git, HF, Zenodo, AuditReport, Rekt, Etherscan); **5 critical-path sources + DISL (3:1 cap negative pool)** are enabled for Run 11; 12 additive deferred to v2.1
- **Preprocessing** — flatten, two-pass compile, dedup (0.85 threshold), normalize, segment, version-bucket
- **Representation** — graphs (v9 schema: 12-dim features, 14 node types, 12 edge types), tokens, with the active 36-issue pre-Run-8 audit regression test guarding the port from `ml/`; **schema-dim gate test** (x.shape[-1] == 12)
- **Labeling** — **5 critical-path crosswalk YAMLs + 5 critical-path parsers** + merger with 99% DoS↔Reentrancy de-duplication + **CallToUnknown < 300 verified → merge into ExternalBug (human-checked)** + **FORGE 50-entry agreement test** + **Go/No-Go minimum-viable-corpus gate**
- **Verification** — the BCCC-failure catcher; AST-level semantic checks + tool corroboration + per-stage p5_s1→p5_s6 regression test + **SmartBugs Curated 143-contract recall test (≥90%)**
- **Splitting + Registry** — versioned train/val/test splits + SQLite catalog with schema migrations + dataset version retirement chain + **NonVulnerable 3:1 cap** (stratified subsample)
- **Analysis** — `complexity_proxy_risk.md` is the data-side L4 finding catcher; co-occurrence matrix flags the 99% BCCC pattern
- **Export + Seam Swap** — sharded export to `sentinel-ml`; predictor.py tier-threshold fix + EMITS edge fix + **slither transitive dep test**
- **Run 11 launch** — first training run on the v2 corpus (5 critical-path sources + DISL negatives, ~4,800 contracts), 2026-08-18

**The 8 most important tests** (the structural defense against the BCCC class of failure):
1. 36-issue pre-Run-8 audit regression test (Stage 2) — every A1–A38 fix is preserved through the port
2. **Schema-dim gate test** (Stage 2/7) — `x.shape[-1] == NODE_FEATURE_DIM` (12, not 11)
3. BCCC Phase 5 regression test (Stage 4) — new module's verification matches the 5-phase deep-dive report to within ±0.5%
4. **SmartBugs Curated 143-contract recall test** (Stage 4) — semantic_checker retains ≥90% of confirmed positives (independent falsification)
5. **Go/No-Go minimum-viable-corpus gate** (Stage 3) — all 6 criteria met OR Run 11 deferred to v2.1 (Run 12)
6. Dual-path seam swap test (Stage 7) — old `dual_path_dataset.py` = new `sentinel_dataset.py` byte-identical
7. **Slither transitive dep test** (Stage 7) — `pip install sentinel-data` brings slither-analyzer for the inference path
8. 7 v2-readiness gates (Stage 7) — schema + Phase 5 + round-trip + complexity + per-class + leakage + 36-issue

---

## Critical context from MEMORY

This folder is the *forward-looking* design. The *historical context* is in:

- **`~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md`** — overall project state, model architecture, current Run 9 results, operational facts
- **`~/.claude/projects/.../memory/project_run8_audit_findings.md`** — the Run 8 audit that motivated this v2 build (89% Reentrancy FP, 86.9% CallToUnknown FP, test_contracts OOD, predictor tier bug)
- **`~/.claude/projects/.../memory/project_data_pipeline_audit.md`** — the data pipeline audit (38.8% BCCC duplication, 99% DoS↔Reentrancy co-occurrence, dedup map by name bug, return_ignored always-0 bug)
- **`~/.claude/projects/.../memory/project_run9_resume.md`** — Run 9 silent-overwrite incident (ep14 best F1=0.2586 lost to ep1 save; lambda typo 0.0075 vs 0.005); the v2 build's watcher F1>0.1 floor is the structural fix
- **`~/.claude/projects/.../memory/reference_solc.md`** — 98 solc versions pre-installed in `~/.solc-select/artifacts/`
- **`~/.claude/projects/.../memory/project_v8_schema.md`** — schema reference (NB: the live schema is v9, not v8 as this file says; verified in the integration proposal §2)

**Read these before starting Stage 0.**

---

## Build window

- **Start:** 2026-06-09
- **End (Stage 7):** 2026-08-17
- **Run 11 launch:** 2026-08-18
- **Total:** ~10 weeks (extended from 8 weeks because of friend's 5 new sources)

---

## Open questions blocking the build (need Ali's nod)

See `actionable_plans/00_INDEX.md` §"Open questions" for the full list. The most urgent:

1. How is `sentinel-data` distributed to `sentinel-ml`? (path dep / PyPI / git tag)
2. Dockerfile base image (recommend `python:3.12.1-bookworm`)
3. ~~DVC remote backend (S3 / GCS / local-only)~~ ✅ **RESOLVED 2026-06-09: local-only for v2 build**
4. Run 11 launch date (recommend 2026-08-18)
5. Should `pdg_builder.py` ship in v2 or defer to v3.1? (current plan: defer)
6. Should `_add_icfg_edges` cross-function external calls be added in Stage 2 (port) or post-Run-11 (v2.1)? (current plan: preserve partial fix)
7. **NEW (friend review):** Critical-path corpus definition (5 sources + DISL negatives) — Run 11 ships with this
8. **NEW (friend review):** NonVulnerable 3:1 cap (default) and CallToUnknown < 300 merge rule (human-checked) — both are config-driven
9. **NEW (friend review):** If minimum-viable-corpus gate fails, defer Run 11 to v2.1 (Run 12) — decision is human-checked at Stage 3 exit

---

**End of README. Start with the binding proposal + the master plan.**
