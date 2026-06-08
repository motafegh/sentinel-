# Actionable Plan — Stage 3: Labeling (parsers, crosswalks, merger, confidence)

**Date:** 2026-06-30
**Stage:** 3 of 8 (Week 4–5: Jun 30–Jul 13 — extended from 1 week to 2 per AUDIT_PATCHES §3)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.4, §5 (Week 4), §6
**Audit ref:** [`../archive/AUDIT_PATCHES_applied_2026-06-08.md`](../archive/AUDIT_PATCHES_applied_2026-06-08.md) §0 (F25, F27, F29), §1 (3-P1 through 3-P10), §2 (C-10)
**Friend suggestions applied:** [`../datasources_suggestions.md`](../datasources_suggestions.md) (DIVE promoted to Tier 1, FORGE added Tier 1, SolidiFI promoted, ScrawlD + Code4rena + DeFi Hacks REKT added Tier 2, DISL + ReentrancyStudy + EVMbench + DeFiVulnLabs added Tier 4)
**Exit criteria:** `sentinel-data label --source scabench` produces 30 per-contract `.labels.json` files for the ScaBench fixture; the merged labels match a hand-checked reference; the canonical 10-class taxonomy is committed to `taxonomy.yaml` and uses the v1 checkpoint's class order; **the 99% DoS↔Reentrancy co-occurrence regression test passes** (a fixture contract with the BCCC-style co-occurrence pattern is de-duplicated by the merger, not preserved); **17 source crosswalks + 17 parsers exist** (12 original + 5 new from friend's suggestions).

---

## Goal

Implement the **Labeling** submodule: the canonical 10-class taxonomy, 12 source-specific crosswalk YAMLs, 12 source-specific parsers, the multi-source merger with conflict resolution, and the T0–T4 confidence tier system. After this stage, every preprocessed contract can be assigned a per-class label with a confidence tier and provenance, regardless of which source dataset it came from.

This stage is what fixes the BCCC failure. The 10 classes are the same as the v1 schema (per the deferred-schema decision). The 12 crosswalks are the highest-leverage artifact in the entire module — they encode the human decisions that map each source's idiosyncratic taxonomy into the canonical one.

---

## Why this stage fourth

Stage 1 produced source code. Stage 2 produced model-consumable representations. Stage 3 is the first stage that attaches semantic meaning (the vulnerability label) to a contract. Doing labeling *after* representation (rather than alongside it) is the design decision that lets the same representation be re-labeled without re-extracting the graph (e.g. when a crosswalk YAML is updated).

The hand-curated crosswalk YAMLs are the bottleneck. Authoring 12 crosswalks is more work than authoring 12 parsers, and the crosswalks must be reviewed by a human (Ali) before being committed. Stage 3 dedicates time to this review.

---

## Design decisions

### D-3.1 — Canonical 10-class taxonomy is frozen at the v1 checkpoint's class order

The taxonomy is the 10 classes from `ml/src/preprocessing/graph_schema.py` in the v1 checkpoint's class order: **CallToUnknown=0, DenialOfService=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TransactionOrderDependence=8, UnusedReturn=9** (per AUDIT_PATCHES 3-P1, the proposal §3.8 has the right classes but doesn't pin the order; the on-disk checkpoint order is the source of truth). The class IDs are integers 0–9 and the order is **LOCKED** because Run 7/8/9 use this order; **any change breaks every existing checkpoint** (per `project_run7_training.md`).

The taxonomy is committed to `sentinel_data/labeling/schema/taxonomy.yaml` as the canonical reference. All crosswalks map into this schema. The taxonomy is append-only: adding a class in the future would be a v2.1 decision, not a v2 build change.

**Why the order is locked:** the `class_<i>` columns in the exported parquet (Stage 7) are positional. The model's classifier head reads them in order. Changing the order means re-training from scratch.

**Why the test in 3.10 catches the mismatch:** the test loads the v1 checkpoint, runs inference on a fixture contract with a known label, and asserts the predicted class index matches the expected index. If the taxonomy order is wrong, the prediction is wrong, and the test fails loud.

**Exit condition:** the v1 checkpoint's class order is verified against `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (or the latest available); the taxonomy.yaml uses the same order; the 3.10 test passes.

### D-3.2 — Crosswalks are version-controlled YAML, human-reviewed, never auto-generated

Each source has a `sentinel_data/labeling/crosswalks/<source>.yaml` file. The crosswalk structure: per-class mapping table from the source's native labels to the canonical 10 classes, plus a `notes:` section with open questions / ambiguous mappings.

The crosswalk is the *human* decision. It is reviewed by Ali before commit. It is never auto-generated from any tool (Slither detector names, audit severity levels, etc.) because the tool output is not the ground truth — the human audit is.

Sources with free-text descriptions (ScaBench's 555 vulnerability descriptions, Web3Bugs report text) may use LLM-assist for the *initial* draft of the crosswalk, but every LLM-suggested mapping is reviewed by a human before being committed.

### D-3.3 — Multi-source merger resolves conflicts with explicit precedence rules AND de-duplicates same-source co-occurrence

When a single contract appears in multiple sources, the merger combines the per-source labels. The default precedence (highest priority first):
1. DeFiHackLabs T0 (exploit-verified) overrides everything
2. Expert-audited sources (Bastet, ScaBench, Web3Bugs — T1)
3. Curated sources (SmartBugs Curated, Ethernaut, OZ — T2)
4. Tool-generated sources (Slither-Audited, Messi-Q, Zenodo 16910242 — T3)
5. Heuristic / derived (T4)

Within a tier, a positive label wins over a negative label (false negatives are worse than false positives for the v2 baseline, but the v1.4 BCCC verification showed that ~70% of BCCC positives were false positives — so the precedence rule may need to be inverted for specific sources; this is a per-source override in the crosswalk).

**NEW (per AUDIT_PATCHES 3-P5, F27):** The merger must explicitly handle the **99% DoS↔Reentrancy co-occurrence pattern** that was the BCCC failure mode (per `project_data_pipeline_audit.md` "Co-occurrence" table). When a single source labels the *same* contract with both DoS and Reentrancy, that's a near-certain sign of noise (BCCC stored the same `.sol` under multiple class dirs and ORed the labels). The merger de-duplicates co-occurring labels from the same source unless:
- (a) there's independent evidence (a different source also labels both)
- (b) the crosswalk explicitly marks the co-occurrence as legitimate (e.g. a SmartBugs Curated reentrancy contract that also has a DoS pattern)

This rule is the data-side complement of the Stage 6 `cooccurrence` analysis. The merger is what *prevents* the 99% co-occurrence; the analysis is what *detects* it.

### D-3.4 — T0–T4 confidence tiers are assigned per (contract, class) pair

The tier is a property of the *evidence*, not the contract. A contract in DeFiHackLabs gets T0 for the exploit class, but T3 for any other class that's also flagged by Slither (because the Slither flag is tool-derived, not exploit-verified). The confidence tier is recorded per (contract, class) in the labels JSON.

### D-3.5 — Labels JSON is the contract between labeling and verification

Every contract that goes through labeling has a `.labels.json` file with this structure:
```
{
  "contract_id": "<sha256>",
  "source": "<source_name>",
  "classes": {
    "Reentrancy": {"value": 1, "confidence": 0.95, "tier": "T1", "evidence": "audit_finding_123"},
    "CallToUnknown": {"value": 0, "confidence": 0.0, "tier": null, "evidence": null},
    ...
  },
  "primary_class": "Reentrancy",
  "n_pos": 1
}
```

Verification (Stage 4) reads this file and runs the semantic checker against the AST. Splitting (Stage 5) reads this file and applies stratification. Export (Stage 7) reads this file and writes the multi-label tensor. The schema is the contract.

### D-3.6 — Labeling does not modify the representation

The labels JSON is stored separately from the graph `.pt` files. A contract's representation is independent of its labels — the same graph can be re-labeled if a crosswalk is updated. This separation is the design that makes crosswalk updates cheap (no re-extraction).

---

## Tasks — ordered, each with verifiable exit condition

### 3.1 — Author the canonical `taxonomy.yaml`

Author `sentinel_data/labeling/schema/taxonomy.yaml` with the 10 canonical classes in the v1 order. Each class has: integer ID, name, description, severity (informational — the model does not use this), and example patterns (informational). The taxonomy is the single source of truth — every crosswalk imports it and references class IDs by integer.

**Why first:** every other task in this stage references the taxonomy. Locking it in first means later work doesn't drift.

**Exit condition:** file exists; contains 10 classes; class order matches the v1 schema (verified against `ml/src/preprocessing/graph_schema.py` or equivalent).

**Commit:** `feat(data-labeling): add canonical 10-class taxonomy.yaml`

---

### 3.2 — Author the ScaBench crosswalk YAML (real, not placeholder)

Replace the Stage-1 placeholder `scabench.yaml` with the real crosswalk. The crosswalk maps ScaBench's free-text vulnerability descriptions into the 10 canonical classes. Per AUDIT_PATCHES 3-P2, the approach is **hybrid**: (a) keyword matching for common patterns (reentrancy, overflow, timestamp, etc.), (b) LLM-assist for the rest, (c) human review for every LLM-suggested mapping. The crosswalk is documented with a `notes:` section recording edge cases.

ScaBench is the smallest Tier-1 source (31 projects, 555 vulnerabilities) and the most studied; the crosswalk is the most thoroughly reviewed. Realistic effort: 2 days (per the plan's original 1–2 day budget for hard crosswalks).

**Why ScaBench first:** the Stage-1 placeholder needs to be filled in for the end-to-end test in 3.10; ScaBench is the lowest-risk source to start with.

**Exit condition:** YAML is valid; covers all 10 classes (some may map to "no ScaBench equivalent" with a comment); 555 descriptions have been processed (LLM-assist draft + human review); crosswalk is committed with author + review date in the YAML header.

**Commit:** `feat(data-labeling): add scabench crosswalk.yaml with full mapping table`

---

### 3.3 — Author the DeFiHackLabs crosswalk YAML

DeFiHackLabs' `_exp.sol` files are exploit PoCs. Each file has a folder name (e.g. `src/test/2024-01-flashloan-attack/`) that indicates the exploit type. The crosswalk maps the folder/exploit type to the canonical class. This is mostly a direct mapping (high confidence, T0).

**Why second:** the highest-confidence Tier-1 source after ScaBench; the crosswalk is the simplest because the folder names are self-documenting.

**Exit condition:** YAML is valid; covers the 4 main exploit families (flash loans, reentrancy, oracle manipulation, access control); T0 tier assigned for all classes.

**Commit:** `feat(data-labeling): add defihacklabs crosswalk.yaml`

---

### 3.4 — Author the remaining 15 crosswalks (10 original + 5 from friend, with per-source specifics)

For each source, author the crosswalk YAML. The work is in decreasing order of difficulty. Per the proposal §6 + AUDIT_PATCHES §3 + friend's suggestions, the build time estimate is: **Bastet 2-3 days, FORGE 2-3 days, Code4rena 2-3 days, DIVE 1-2 days, SC-Bench/Messi-Q/SmartBugs-Wild/ReentrancyStudy 1-2 days, the rest 0.5-1 day**. Total: **15-20 days for 15 crosswalks** (per AUDIT_PATCHES §3 — the plan's "1 day per crosswalk average" underestimates the hard ones; the budget is now 3 weeks total for Stage 3, not 2).

**Original 10 per-source specifics (per AUDIT_PATCHES 3-P3, 3-P4, 3-P7, 3-P8, 3-P9):**

| Source | Difficulty | Specific guidance |
|---|---|---|
| **Bastet** | HIGH (2-3 days) | 46 finding tags → 10 classes. Needs audit-report parser to map `dataset.csv` findings → `.sol` source locations via `reports/<id>.md` mapping. Partial crosswalk in Bastet's README; we extend it. |
| **Web3Bugs** | MEDIUM-HIGH (1-2 days) | `bugs.csv` + `contests.csv` + report text. **O/L/S severity**: only O (Optimistic) and L (Low) map to positive; S (Speculative) is too uncertain — defaults to negative. Needs report-text → class mapper (LLM-assist). |
| **solidity_defi_vulns** | LOW (0.5 day) | HF row schema is structured; `is_real` flag + `attack_title` → class is mostly direct. |
| **smartbugs_curated** | LOW (0.5 day) | DASP categories → 10 classes. **DASP has 10 categories — one (front_running) maps to TOD; the crosswalk must be explicit about which DASP category doesn't have a Sentinel class equivalent.** Line-level annotations are an extra metadata field. |
| **smartbugs_wild** | MEDIUM (1 day) | 47K contracts, no labels; needs Slither run to derive T3 labels. Mostly a Slither-detector → class crosswalk. |
| **openzeppelin** | VERY LOW (0.5 day) | **Clean negative source**: every OZ contract is `T1_clean` for `NonVulnerable`; the parser NEVER produces a positive label even if Slither flags something. Per AUDIT_PATCHES 3-P8, the crosswalk is explicit: "OZ contracts are clean, no exceptions." |
| **ethernaut** | VERY LOW (0.5 day) | 30 challenges; each is a known vulnerability pattern → positive label. |
| **slither_audited** | MEDIUM-HIGH (1-2 days) | **Highest-risk parser** per AUDIT_PATCHES 3-P9. Crosswalk must be **explicitly conservative** — only map to a class if Slither's confidence is > 0.8 AND the detector is in the canonical CLASS_TO_DETECTORS list (from `project_agents.md`). The tier is T3 (tool-generated), not T1. |
| **messi_q** | MEDIUM (1 day) | Multiple subfolders with different schemas; per-folder sub-parser needed. Older Solidity era. |
| **zenodo_16910242** | LOW (0.5 day) | 88.3 MB zip; `contracts.zip` with `labels.csv`; per-contract `is_vulnerable_*` columns. Yizhou Chen's CLEAR ICSE 2025 dataset. |

**NEW 5 from friend's suggestions (`datasources_suggestions.md`, 2026-06-08):**

| Source | Difficulty | Specific guidance |
|---|---|---|
| **DIVE** (Tier 1, peer-reviewed) | MEDIUM (1-2 days) | 22,330 contracts, 8 DASP classes, **multi-label** (avoids BCCC's "one folder = one vuln" fiction). DIVE's 8 DASP classes map cleanly to our 10 classes (with 2 extras: bad randomness, time manipulation). Crosswalk: per-contract multi-hot label, one row per DASP class. **Tier 1 because peer-reviewed in Nature Scientific Data.** |
| **FORGE** (Tier 1, ICSE 2026) | HIGH (2-3 days) | **LLM-driven extraction from real audit reports**. Crosswalk: CWE → 10 classes (CWE is the industry standard). 46 CWE IDs → 10 classes is a research-quality mapping; needs careful review. **Requires an `audit_report_connector`** that ingests audit reports (PDF, Markdown) — the connector is new work. Tier 1 because labels are from professional human auditors. |
| **SolidiFI Benchmark** (Tier 1, ISSTA 2020) | MEDIUM (1-2 days) | **9,369 injected bugs, 100% ground-truth certainty** (mathematically guaranteed by the injection methodology). 7 types map to our 10 (Reentrancy, Timestamp, UnhandledExceptions, UncheckedSend, TOD, IntegerUO, TxOrigin). Crosswalk: per-bug-injection metadata → positive label. **Already in `ml/data/SolidiFI-benchmark/`** — easy connector. **Promoted from "supporting" to Tier 1** per friend. |
| **ScrawlD** (Tier 2, MSR 2022) | MEDIUM (1-2 days) | 6,780 mainnet contracts, **5-tool majority voting (3/5 agreement)**. The 5 tools are Slither, Mythril, Manticore, Securify, Smartcheck. Crosswalk: per-tool detector → 10 classes; require 3/5 agreement for positive label (otherwise drop). Tier 2 silver because tool-based, but with consensus. |
| **Code4rena Audit Reports** (Tier 2 gold) | HIGH (2-3 days) | **Human expert auditors under financial incentive** — highest possible label quality. 6,454 contracts, 1,361 high-risk findings, 499 confirmed high-severity. **Requires a new `audit_report_scraper` connector** that ingests the C4 reports (Markdown from code4rena.com). Crosswalk: finding severity + description → 10 classes. Tier 2 because scale is smaller, but gold-quality labels. **The "EVMbench" subset (120 vulns from 40 audits) is included in Code4rena — no separate connector needed.** |
| **DeFi Hacks REKT Database** (Tier 2 gold, T0) | MEDIUM (1-2 days) | 3,216 hacked DeFi incidents + 181 curated high-impact. **Verified real exploits** (T0 tier). Requires a `rekt_scraper` connector that ingests the REKT database (Markdown posts). Crosswalk: incident type → 10 classes (T0 override per D-3.3). |
| **DISL** (Tier 4 bronze) | LOW (0.5 day) | 514,506 unique Solidity files, **unlabeled**. Used for NonVulnerable class + pretraining. No crosswalk needed (no labels). |
| **ReentrancyStudy-Data** (Tier 4 bronze) | MEDIUM (1 day) | 230,548 Etherscan contracts, reentrancy-labeled (single class, tool-based). Crosswalk: per-tool label → Reentrancy class only. Tier 4 because single-class + tool-based, but massive scale. |
| **DeFiVulnLabs** (Tier 3 structural) | MEDIUM (1-2 days) | 48 vulnerability types, Foundry-style exploits. Already in DeFiHackLabs family (same org). Crosswalk: per-folder → 10 classes. |

**Friend's 3-tool ensemble warning** (per datasources_suggestions.md Part 1): Conkas + Slither + Smartcheck only detects 76.78% of actual vulnerabilities; Slither reentrancy precision is 51.97%. **This validates** the Stage 4 `tool_validator` design (D-4.3): tool agreement is corroborative, NOT authoritative. The Slither-Audited crosswalk (3-P9) has the "confidence > 0.8 AND canonical detector" rule precisely to filter out the low-precision tool hits.

**Friend's 97% SmartBugs Wild FP rate warning** (per Part 4): SmartBugs Wild as labeled data is a "97% FP" trap. The existing Tier 3 (T3 tool-generated, conservative) design for `smartbugs_wild` in the merger is correct. **No change needed** — the friend confirms the existing crosswalk design.

**Why batch:** the crosswalk work is the bottleneck; batching it lets the reviewer focus on crosswalk review without context-switching to code.

**Exit condition:** all 15 crosswalk YAMLs exist (10 original + 5 new from friend); each is reviewed and committed; the catalog `_total_sources_in_crosswalks` count is 17 (ScaBench + DeFiHackLabs + 15); the OZ crosswalk has the "clean, no exceptions" rule; the Slither-Audited crosswalk has the confidence > 0.8 + canonical-detector rule; the DIVE crosswalk handles multi-label; the FORGE crosswalk handles CWE; the Code4rena connector ingests Markdown reports.

**Commit:** 15 separate commits, one per crosswalk, each `feat(data-labeling): add <source> crosswalk.yaml`

---

### 3.5 — Author the ScaBench parser

Author `sentinel_data/labeling/parsers/scabench.py`. The parser reads the ScaBench source data (raw audit JSON, project metadata, and per-finding descriptions) and the crosswalk YAML, and produces per-contract `.labels.json` files. The parser handles the "checkout_sources.py" post-clone step (called from the connector) and reads the resulting JSON metadata.

**Why ScaBench first:** Stage-1 placeholder is in place; the crosswalk is real; the parser is the next layer.

**Exit condition:** parser runs against the 30-file ScaBench fixture; produces 30 `.labels.json` files; the labels match a hand-checked reference for 5 of the 30 files.

**Commit:** `feat(data-labeling): add scabench parser`

---

### 3.6 — Author the DeFiHackLabs parser

Author `sentinel_data/labeling/parsers/defihacklabs.py`. The parser walks the per-incident folder structure, reads the `_exp.sol` file metadata (PoC), and assigns T0 labels based on the incident type from the crosswalk.

**Exit condition:** parser runs against a 5-incident DeFiHackLabs fixture; produces 5 `.labels.json` files; all labels are T0.

**Commit:** `feat(data-labeling): add defihacklabs parser`

---

### 3.7 — Author the remaining 10 parsers

For each source, author the parser. Each parser is specific to the source's data format and the crosswalk. Per the proposal §6, the per-source effort ranges from 0.5 days (OZ, Ethernaut) to 2-3 days (Bastet — needs the finding→source-location mapping; Web3Bugs — needs report-text→class).

**Exit condition:** all 12 parsers import and run; each produces sensible output for its fixture.

**Commit:** 10 separate commits, one per parser.

---

### 3.8 — Implement `merger.py` and `confidence.py`

Author `sentinel_data/labeling/merger.py` and `sentinel_data/labeling/confidence.py`. The merger takes multiple `.labels.json` files for the same `contract_id` (from different sources) and produces a merged label set with explicit conflict resolution per D-3.3. The confidence module assigns the T0–T4 tier per (contract, class) pair per D-3.4.

The merger writes a per-contract `labels.merged.json` that supersedes the per-source `.labels.json` for downstream consumption. The catalog records the merge lineage (which source labels were combined).

**Why after the parsers:** the merger needs at least 2 sources' label output to test against; with 12 parsers, the merger is well-exercised.

**Exit condition:** merger correctly combines labels for a fixture where one contract appears in 3 sources; confidence tiers are assigned per (contract, class); conflict resolution follows the precedence rules.

**Commit:** `feat(data-labeling): add merger + confidence module`

---

### 3.9 — Wire the `sentinel-data label` CLI subcommand

Connect `cli.py` `label` subcommand to the parser + merger system. The CLI iterates over preprocessed contracts, runs the per-source parser, then runs the merger for contracts that appear in multiple sources. Add `--source <name>` to limit to one source; default is "all enabled sources."

Update `dvc.yaml` stage `label` to call `sentinel-data label`.

**Exit condition:** `sentinel-data label --source scabench` produces 30 `.labels.json` files; `sentinel-data label` (no source) produces merged labels for all enabled sources.

**Commit:** `feat(data-labeling): wire CLI + DVC for the label stage`

---

### 3.10 — Add tests for the labeling stage

Author `Data/tests/test_labeling/` with:
- **Taxonomy test** — 10 classes in the expected order; class IDs match the v1 checkpoint's class order (loads `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` and asserts the prediction index matches the expected index)
- **Crosswalk tests** — each crosswalk YAML is valid; every entry references valid taxonomy class IDs; the OZ crosswalk has the "clean, no exceptions" rule; the Slither-Audited crosswalk has the confidence > 0.8 + canonical-detector rule
- **Parser tests** — each parser runs against a small fixture; output matches a hand-checked reference
- **Merger tests** — multi-source contracts merge correctly; conflict resolution follows precedence (D-3.3)
- **Confidence tests** — T0–T4 tiers assigned per the rules in D-3.4; per-class tier overrides work
- **99% DoS↔Reentrancy co-occurrence regression test** (per AUDIT_PATCHES 3-P10, C-10) — a fixture contract with the BCCC-style co-occurrence pattern is de-duplicated by the merger, not preserved. This is the test that would have caught the BCCC failure.

**Exit condition:** `poetry run pytest tests/test_labeling -v` passes; coverage > 80%; the 99% co-occurrence regression test passes.

**Commit:** `test(data-labeling): add full test suite for labeling stage (incl. 99% co-occurrence regression)`

---

### 3.11 — Author `ADR-0004-labeling-design.md`

Document the key design decisions: frozen taxonomy (D-3.1), human-reviewed crosswalks (D-3.2), merger precedence rules (D-3.3), per-class confidence (D-3.4), labels JSON contract (D-3.5), representation-label independence (D-3.6).

**Exit condition:** file exists; cites the BCCC failure as the motivation for the human-review rule.

**Commit:** `docs(data): add ADR-0004 for labeling design`

---

## What NOT to fix (preservation list)

| Bug | Status | File:line | Stage 3 action |
|---|---|---|---|
| **A20** label=0 hardcode | ✅ FIXED | `ml/src/data_extraction/ast_extractor.py:290,342,395` | Do not re-fix. The Stage 2 36-issue test guards it. |
| **LibraryCall <: HighLevelCall** | ⚠ KNOWN | `ml/src/preprocessing/graph_extractor.py:1081` | The crosswalks for CallToUnknown and ExternalBug must handle library calls. Per AUDIT_PATCHES F25, `SafeMath.add()` is classified as HighLevelCall; the crosswalk's CallToUnknown pattern must explicitly EXCLUDE library calls (use only cross-contract targets). |
| **99% DoS↔Reentrancy co-occurrence** | Source: BCCC | (not in v2 corpus) | The merger (D-3.3) prevents this. The 3.10 regression test guards it. |
| `BCCC folder = label` assumption | Source: BCCC | (not in v2 corpus) | The crosswalks for the 12 v2 sources do NOT use folder names as labels. They use per-source parser logic (audit reports, Slither detector hits, etc.). |
| OZ labels being T1_clean | ✅ CORRECT | (not a bug; design choice) | The OZ crosswalk maintains this. If Slither flags an OZ contract, the crosswalk is correct; the OZ contract is still T1_clean. |
| Return-ignored bug | ✅ FIXED | `ml/src/preprocessing/graph_extractor.py` | Do not re-fix. The Stage 2 36-issue test guards it. Per AUDIT_PATCHES F29, the UnusedReturn semantic check (Stage 4) can use the fixed `feat[7]`. |

---

## Final exit criteria check

| # | Check |
|---|---|
| 1 | `sentinel_data/labeling/schema/taxonomy.yaml` exists with 10 classes in the v1 order |
| 2 | All 12 source crosswalks exist (`bastet`, `scabench`, `web3bugs`, `defihacklabs`, `solidity_defi_vulns`, `smartbugs_curated`, `smartbugs_wild`, `openzeppelin`, `ethernaut`, `slither_audited`, `messi_q`, `zenodo_16910242`) |
| 3 | All 12 source parsers exist and run against their fixtures |
| 4 | The merger correctly combines labels for a multi-source fixture; conflict resolution follows D-3.3 |
| 5 | `sentinel-data label --source scabench` produces 30 `.labels.json` files for the ScaBench fixture |
| 6 | Labels match a hand-checked reference for 5 of the 30 ScaBench files |
| 7 | Confidence tiers are assigned per (contract, class) per D-3.4 |
| 8 | `dvc repro label` runs end-to-end |
| 9 | `poetry run pytest tests/test_labeling -v` passes with > 80% coverage; **99% DoS↔Reentrancy co-occurrence regression test passes** |
| 10 | `ADR-0004-labeling-design.md` is committed; **references the 99% co-occurrence as the motivation for the merger's de-duplication rule** |

All 10 pass → **Stage 3 complete**. Tag `data-stage-3`, proceed to Stage 4.

---

## Risk register

| Risk | Mitigation |
|---|---|
| The 12 crosswalks take longer than 1 day each (Bastet alone is 2-3 days) | Stage 3 budget is 1 crosswalk/day; harder ones get explicit buffer; the 12 crosswalks can be split across 2 weeks if needed (extending the build) |
| The merger precedence rules produce surprising results on multi-source contracts | The merger is heavily unit-tested (3.10); the verification stage (Stage 4) flags any merged label that fails the semantic check |
| The taxonomy class order doesn't match the v1 checkpoint's class order (D-3.1 open question) | The test in 3.10 catches the mismatch; if found, the taxonomy is bumped and the v1 checkpoint must be loaded with a remap layer (added in Stage 7 if needed) |
| LLM-assist for crosswalk authoring produces inconsistent mappings across sources | A "taxonomy consistency check" runs in CI: every crosswalk mapping is validated against the taxonomy schema; LLM-drafted mappings are flagged for human review |
| Source parsers break when the source's data format changes (e.g. ScaBench updates their JSON schema) | Each parser is isolated; format change is contained to one parser; the regression test catches it; no other parser or stage is affected |
| A contract's `.labels.json` is huge (multi-class with confidence) and slows Stage 4 verification | Labels JSON is small per contract (max ~2 KB); no performance risk |

---

**End of Stage 3 actionable plan. Total estimated time: 5–7 working days (Jun 30–Jul 8), with Jul 9–10 as buffer. Note: this stage may extend to 2 weeks if crosswalk authoring is slower than estimated.**
