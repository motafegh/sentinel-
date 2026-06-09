# Actionable Plan — Stage 3: Labeling (parsers, crosswalks, merger, confidence)

**Date:** 2026-06-30 (revised 2026-06-09 post-friend-review)
**Stage:** 3 of 8 (Week 4–6: Jun 30–Jul 20 — 3 weeks, critical-path budget)
**Owner:** SENTINEL data engineering
**Source proposal:** `docs/proposal/Data_Module_Proposals/Sentinel_v2_Data_Module_Integration_Proposal.md` §3.4, §5 (Week 4–6), §6
**Audit ref:** [`../archive/AUDIT_PATCHES_applied_2026-06-08.md`](../archive/AUDIT_PATCHES_applied_2026-06-08.md) §0 (F25, F27, F29), §1 (3-P1 through 3-P10), §2 (C-10)
**Friend suggestions applied:** [`../datasources_suggestions.md`](../datasources_suggestions.md) (DIVE promoted to Tier 1, FORGE added Tier 1, SolidiFI promoted, ScrawlD + Code4rena + DeFi Hacks REKT added Tier 2, DISL + ReentrancyStudy + EVMbench + DeFiVulnLabs added Tier 4)
**Friend-review revisions (2026-06-09):**
- **Source list: 17 → 5 critical-path + 12 additive** (Run 11 ships with the 5; the 12 are v2.1)
- **5 critical-path:** DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs (+ DISL as NonVulnerable source, no crosswalk)
- **DIVE "bad randomness" dropped** (no 10-class equivalent; per proposal §6.3.3)
- **Code4rena scraper dropped** (Bastet replaces it; per proposal §6.2)
- **ReentrancyStudy dropped entirely** (per proposal §6.3.4)
- **CallToUnknown merge rule added** (human-checked, not auto-merge; per proposal §6.3.2)
- **FORGE 50-entry agreement test added** (if <85%, defer FORGE to v2.2; per proposal §6.5)
- **Go/No-Go minimum-viable-corpus gate added** (if corpus doesn't meet thresholds, defer Run 11; per proposal §6.5)
**Exit criteria:** `sentinel-data label --source defihacklabs` produces 30 per-contract `.labels.json` files for the DeFiHackLabs fixture; the merged labels match a hand-checked reference; the canonical 10-class taxonomy is committed to `taxonomy.yaml` and uses the v1 checkpoint's class order; **the 99% DoS↔Reentrancy co-occurrence regression test passes** (a fixture contract with the BCCC-style co-occurrence pattern is de-duplicated by the merger, not preserved); **5 critical-path crosswalks + 5 critical-path parsers exist**; **DIVE crosswalk drops "bad randomness" with comment**; **FORGE 50-entry agreement test passes (or FORGE is deferred to v2.2)**; **CallToUnknown < 300 verified → merger pauses and asks Ali** (config rule); **Go/No-Go minimum-viable-corpus gate returns 0** (or Run 11 is deferred to v2.1 with documented decision).

## Goal

Implement the **Labeling** submodule: the canonical 10-class taxonomy, **5 critical-path source-specific crosswalk YAMLs** (DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs) + DISL as NonVulnerable source (no crosswalk), **5 critical-path source-specific parsers**, the multi-source merger with conflict resolution, the **CallToUnknown < 300 merge rule** (human-checked), the **FORGE 50-entry agreement test** (if FORGE is added), the T0–T4 confidence tier system, and the **Go/No-Go minimum-viable-corpus gate**. After this stage, every preprocessed contract can be assigned a per-class label with a confidence tier and provenance, regardless of which source dataset it came from.

The 12 additive sources (Bastet, FORGE, ScrawlD, DeFi Hacks REKT, Ethernaut, OZ Contracts, solidity_defi_vulns, DeFiVulnLabs, SC-Bench, SmartBugs Wild, slither-audited, Zenodo 16910242 + Messi-Q + ScaBench) are **deferred to v2.1**. The Stage 3 plan authorises 1-2 additional crosswalks beyond the 5 critical-path if the 3-week budget permits (Bastet is the first candidate; Code4rena scraper is removed entirely — Bastet replaces it).

This stage is what fixes the BCCC failure. The 10 classes are the same as the v1 schema (per the deferred-schema decision). The 5 critical-path crosswalks are the highest-leverage artifact in the entire module — they encode the human decisions that map each source's idiosyncratic taxonomy into the canonical one.

---

## Why this stage fourth

Stage 1 produced source code. Stage 2 produced model-consumable representations. Stage 3 is the first stage that attaches semantic meaning (the vulnerability label) to a contract. Doing labeling *after* representation (rather than alongside it) is the design decision that lets the same representation be re-labeled without re-extracting the graph (e.g. when a crosswalk YAML is updated).

The hand-curated crosswalk YAMLs are the bottleneck. Authoring 5 critical-path crosswalks is more work than 5 parsers, and the crosswalks must be reviewed by a human (Ali) before being committed. Stage 3 dedicates time to this review. The 12 additive crosswalks are out of scope for v2 — they are v2.1 work.

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

### 3.4 — Author the 4 remaining critical-path crosswalks (DeFiHackLabs done in 3.3)

The ScaBench and DeFiHackLabs crosswalks are done in 3.2 and 3.3. The remaining 3 critical-path crosswalks are SolidiFI, DIVE, Web3Bugs. (SmartBugs Curated's crosswalk is a one-liner — DASP → 10 classes direct, see §6.4 / config.yaml `smartbugs_curated` — and is committed in this same task.)

For each source, author the crosswalk YAML. The work is in decreasing order of difficulty. Per the proposal §6.1 + AUDIT_PATCHES §3 + friend's suggestions, the build time estimate for the 5 critical-path crosswalks is: **DeFiHackLabs 1 day, SolidiFI 1-2 days, DIVE 1-2 days, SmartBugs Curated 0.5 day, Web3Bugs 1-2 days**. Total: **5-8 days for 5 critical-path crosswalks** (within the 3-week Stage 3 budget; the remaining 1.5 weeks are for parsers + merger + tests + agreement test + gate).

**Critical-path 5 per-source specifics (per AUDIT_PATCHES 3-P3, 3-P4, 3-P7, 3-P8, 3-P9 + friend review):**

| Source | Difficulty | Specific guidance |
|---|---|---|
| **DeFiHackLabs** (DONE in 3.3) | LOW (1 day) | Exploit-PoC `_exp.sol` files; folder name = exploit type → class is direct. T0 confidence. |
| **SolidiFI** | MEDIUM (1-2 days) | **9,369 injected bugs, 100% ground-truth certainty** (mathematically guaranteed by the injection methodology). 7 types map to our 10 (Reentrancy, Timestamp, UnhandledExceptions, UncheckedSend, TOD, IntegerUO, TxOrigin). Crosswalk: per-bug-injection metadata → positive label. **Already in `ml/data/SolidiFI-benchmark/`** — easy connector. **Promoted from "supporting" to Tier 1** per friend. |
| **DIVE** | MEDIUM (1-2 days) | 22,330 contracts, 8 DASP classes, **multi-label** (avoids BCCC's "one folder = one vuln" fiction). **DIVE class mapping per proposal §6.4 (and config.yaml `dive_class_mapping`):** reentrancy → Reentrancy, access_control → ExternalBug, arithmetic → IntegerUO, unchecked_low_level_calls → CallToUnknown, denial_of_service → DoS, front_running → TOD, time_manipulation → Timestamp. **"bad_randomness" is DROPPED** (no 10-class equivalent; per proposal §6.3.3 + friend review). Crosswalk YAML has a comment documenting the drop decision for v2.1. |
| **SmartBugs Curated** | LOW (0.5 day) | DASP categories → 10 classes. DASP categories map: reentrancy → Reentrancy, access_control → ExternalBug, arithmetic → IntegerUO, unchecked_low_level_calls → CallToUnknown, denial_of_service → DoS, front_running → TOD, time_manipulation → Timestamp, bad_randomness → DROPPED. Line-level annotations are an extra metadata field. The 143 contracts are the **ground-truth probe for the semantic_checker recall test in Stage 4.11** (per friend review). |
| **Web3Bugs** | MEDIUM-HIGH (1-2 days) | `bugs.csv` + `contests.csv` + report text. **O/L/S severity**: only O (Optimistic) and L (Low) map to positive; S (Speculative) is too uncertain — defaults to negative. Needs report-text → class mapper (LLM-assist). |

**Friend's 3-tool ensemble warning** (per datasources_suggestions.md Part 1): Conkas + Slither + Smartcheck only detects 76.78% of actual vulnerabilities; Slither reentrancy precision is 51.97%. **This validates** the Stage 4 `tool_validator` design (D-4.3): tool agreement is corroborative, NOT authoritative. The 12 additive crosswalks (deferred to v2.1) include Slither-Audited; the conservative "confidence > 0.8 AND canonical detector" rule is preserved for that future work.

**Why batch:** the crosswalk work is the bottleneck; batching it lets the reviewer focus on crosswalk review without context-switching to code.

**Exit condition:** all 5 critical-path crosswalk YAMLs exist (DeFiHackLabs, SolidiFI, DIVE, SmartBugs Curated, Web3Bugs); each is reviewed and committed; the catalog `_total_sources_in_critical_path_crosswalks` count is 5; the DIVE crosswalk has the "bad_randomness: DROPPED" comment with v2.1 migration note; the Web3Bugs crosswalk has the O/L/S severity filter.

**Commit:** 5 separate commits, one per crosswalk, each `feat(data-labeling): add <source> crosswalk.yaml`

**12 additive crosswalks (v2.1, not in this plan):** Bastet, FORGE, ScrawlD, DeFi Hacks REKT, Ethernaut, OZ Contracts, solidity_defi_vulns, DeFiVulnLabs, SC-Bench, SmartBugs Wild, slither-audited, Zenodo 16910242, Messi-Q, ScaBench. See [`actionable_plans/v2_1_additive_labeling.md`](v2_1_additive_labeling.md) (TODO: create when v2.1 starts).

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

**NEW 2026-06-09 (friend review §6.3.2):** the merger **runs the CallToUnknown < 300 verified → ExternalBug merge rule** as a per-class count check after all parsers have run. The rule:

```python
# In merger.py
def check_call_to_unknown_merge_rule(merged_labels: dict) -> dict:
    """Returns a status dict; pauses and asks human if trigger met."""
    call_to_unknown_count = sum(
        1 for lbl in merged_labels.values()
        if lbl.get('classes', {}).get('CallToUnknown', {}).get('value') == 1
    )
    trigger_threshold = 300  # from config.yaml pipeline.class.merge_rules
    if call_to_unknown_count < trigger_threshold:
        return {
            "trigger": True,
            "action": "pause_and_ask_human",
            "current_count": call_to_unknown_count,
            "threshold": trigger_threshold,
            "proposed_change": "labels.class_map.CallToUnknown = ExternalBug",
            "reversible": True,
            "decision": None,  # filled in by human approval
        }
    return {"trigger": False, "current_count": call_to_unknown_count}
```

**The rule pauses, not auto-merges.** Ali's explicit approval is required before the merger applies the merge. The decision is recorded in the catalog's `merge_decisions` table. The merge is reversible in v2.1: if we accumulate 1000+ verified CallToUnknown, split them back.

**Why after the parsers:** the merger needs at least 2 sources' label output to test against; with 5 critical-path parsers, the merger is well-exercised.

**Exit condition:** merger correctly combines labels for a fixture where one contract appears in 3 sources; confidence tiers are assigned per (contract, class); conflict resolution follows the precedence rules; **CallToUnknown < 300 rule pauses and asks human (not auto-merge)**.

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

### 3.12 — NEW 2026-06-09 (friend review): FORGE 50-entry agreement test (IF FORGE is added)

If the additive list includes FORGE (i.e. Stage 3 has budget for 1-2 additive crosswalks beyond the 5 critical-path), run a 50-entry agreement test on FORGE's LLM-driven label extraction. The test:

1. **Sample 50 entries** from FORGE's raw label file (stratified by CWE class).
2. **Manually extract** the code location + Sentinel class for each (by Ali, with the 46-CWE → 10-class crosswalk as a reference).
3. **Compare** the LLM-extracted label to the manual extraction. Agreement = (matches) / 50.
4. **Threshold:** if agreement ≥ 85% (per `pipeline.min_viable_corpus.forge_agreement_min` in config.yaml), FORGE is added to the additive list and ships in Run 11.
5. **If < 85%:** FORGE is **deferred to v2.2** (per proposal §6.5). The 5 critical-path sources ship without it. The decision is documented in `data/registry/decisions/forge_50_entry_test.md`.

**Why 85%:** the friend recommendation. A 100% LLM extraction is unrealistic (LLMs are noisy on long technical documents); 85% is the "good enough" bar where LLM errors are within human-reviewable range. Below 85%, the LLM extraction path is not production-ready.

**Why 50 entries:** stratified by CWE class covers the 46 CWE IDs; 50 is statistically significant at the 85% threshold (95% CI of ±10%). Larger samples are incremental precision.

**Why this is conditional:** if Stage 3 has time for FORGE, it ships. If not, the 5 critical-path sources ship without it, and FORGE is v2.1.

**Exit condition:** if FORGE is added: 50-entry test passes ≥85% agreement; test result is committed to `data/verification/forge_50_entry_test/agreement.json`. If FORGE is not added: decision is documented in `data/registry/decisions/forge_deferred_to_v2_2.md`.

**Commit:** `test(data-labeling): add FORGE 50-entry agreement test (or document deferral)`

---

### 3.13 — NEW 2026-06-09 (friend review): Go/No-Go minimum-viable-corpus gate

At the end of Stage 3, run the **minimum-viable-corpus gate** (per proposal §6.5) before transitioning to Stage 4. The gate validates that the 5 critical-path sources + DISL negatives hit the minimum thresholds:

| # | Criterion | Threshold (from config.yaml `pipeline.min_viable_corpus.*`) | If below |
|---|---|---|---|
| 1 | Total contracts (5 critical-path + DISL negatives) | ≥ 4,000 (`total_contracts_min`) | Defer Run 11 to v2.1 (Run 12) |
| 2 | Reentrancy, DoS, IntegerUO positive count | ≥ 300 each (`per_class_positive_min_major`) | Defer Run 11 |
| 3 | Other 7 classes positive count | ≥ 100 each (`per_class_positive_min_minor`) | Defer Run 11 OR apply CallToUnknown merge rule |
| 4 | CallToUnknown verified count | ≥ 300 (`call_to_unknown_min`) | Apply merge rule (3.8) — NOT a defer trigger |
| 5 | SmartBugs Curated semantic_checker recall | ≥ 90% (`smartbugs_curated_recall_min`) | Defer Run 11 (semantic_checker is broken) |
| 6 | FORGE agreement (if FORGE added) | ≥ 85% (`forge_agreement_min`) | Defer FORGE to v2.2 — NOT a defer trigger for Run 11 |

**The gate is automated:** `sentinel-data verify --min-viable-corpus` exits 0 if all 6 criteria are met, non-zero otherwise. The gate output is a per-criterion pass/fail with the actual count vs threshold.

**If the gate fails (criteria 1-3 or 5):**
- **The decision is documented, not auto-deferred.** Ali reviews the gate output and decides: (a) defer Run 11 to v2.1 (Run 12), OR (b) attempt fast re-introduction of 1-2 additive sources to fill the gap (e.g. add Bastet if Reentrancy < 300; add OZ Contracts if NonVulnerable is below 3:1 cap).
- The decision is committed to `data/registry/decisions/stage3_min_viable_corpus_gate.md`.
- The whole 10-week timeline is re-baselined if Run 11 is deferred.

**If the gate passes:** Stage 3 is complete; proceed to Stage 4.

**Exit condition:** the gate command runs cleanly; the gate output is committed; Ali's decision is recorded; **Run 11 launch is either confirmed or deferred to v2.1 (Run 12) with documented reason**.

**Commit:** `feat(data-labeling): add min_viable_corpus gate (Stage 3 exit criterion)`

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
| 2 | All 5 critical-path source crosswalks exist (`defihacklabs`, `solidifi`, `dive`, `smartbugs_curated`, `web3bugs`); the DIVE crosswalk has the "bad_randomness: DROPPED" comment with v2.1 migration note |
| 3 | All 5 critical-path source parsers exist and run against their fixtures |
| 4 | The merger correctly combines labels for a multi-source fixture; conflict resolution follows D-3.3; **CallToUnknown < 300 merge rule pauses and asks human (not auto-merge)** |
| 5 | `sentinel-data label --source defihacklabs` produces 30 `.labels.json` files for the DeFiHackLabs fixture |
| 6 | Labels match a hand-checked reference for 5 of the 30 DeFiHackLabs files |
| 7 | Confidence tiers are assigned per (contract, class) per D-3.4 |
| 8 | `dvc repro label` runs end-to-end |
| 9 | `poetry run pytest tests/test_labeling -v` passes with > 80% coverage; **99% DoS↔Reentrancy co-occurrence regression test passes** |
| 10 | `ADR-0004-labeling-design.md` is committed; references the 99% co-occurrence as the motivation for the merger's de-duplication rule |
| 11 | **(NEW) FORGE 50-entry agreement test** (if FORGE added): ≥85% agreement OR FORGE deferred to v2.2 with documented decision |
| 12 | **(NEW) Go/No-Go minimum-viable-corpus gate**: all 6 criteria met OR Run 11 deferred to v2.1 with documented decision |

All 12 pass → **Stage 3 complete**. Tag `data-stage-3`, proceed to Stage 4.

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

**End of Stage 3 actionable plan. Total estimated time: 5–8 working days for the 5 critical-path crosswalks + 5–7 days for parsers + merger + tests + agreement test + gate = 10–15 working days (Jun 30–Jul 18), with Jul 19–20 as buffer for the gate decision. Note: this stage may extend to the full 3 weeks if 1-2 additive crosswalks are added (e.g. Bastet for scale).**
