# Data Source Addition Plan — v4+
**Date:** 2026-06-13
**Author:** Ali + Claude
**Purpose:** Comprehensive plan for adding new vulnerability data sources to the SENTINEL data module without breaking the v3 baseline that Run 12 will train on.
**Parent plan:** `data_module/temp/pre-run12-fixes-2026-06-13.md` (Step A done, Step C done, Step B+D pending)
**Status:** Design phase. No code changes yet.

---

## Context

### Current state of the v2 data module (post-Stage 7B + post-SmartBugs ingestion)

| Component | State |
|---|---|
| **Active sources** | DIVE (22,073, T2), SolidiFI (283, T0), SmartBugs Curated (137 of 143, T1) |
| **Total preprocessed** | 22,493 contracts |
| **Total with representations** | 21,657 (graph + tokens) |
| **V3 export** | `sentinel-v3-smartbugs-2026-06-13` — 5 shards, hash-verified |
| **V3 splits** | 18,559 train / 2,009 val / 1,925 test — 0% cross-split leakage |
| **Schema** | v9 (locked, 12 features, 14 node types, 12 edge types) |
| **Taxonomy** | 10 classes (per `taxonomy.yaml`, LABELING order) |
| **NUM_CLASSES** | 10 (hard-coded in `trainer.py` classifier head) |
| **Seam-swap** | Done (sentinel_data is source of truth, ml is consumer) |

### What this plan covers

The pre-run12-fixes plan added ONE new source (SmartBugs) and got us to a v3 export. **This plan covers the next 2-4 data sources** we want to add, with the explicit goal of:

1. **Not breaking** the v3 baseline that Run 12 will use
2. **Not corrupting** any of the 36-issue pre-Run-8 audit regression tests
3. **Not changing** the locked schema (v9), taxonomy (10 classes), or export format (v1)
4. **Following the same shape** the SmartBugs ingestion used (crosswalk + parser + pipeline + export)

### What this plan does NOT cover

- Run 12 launch and training (see pre-run12-fixes Step D)
- Level-3 dedup (see pre-run12-fixes Step B)
- Stage 5 registry (August project, not a pre-Run-12 item)
- DeFiHackLabs (BLOCKED in pre-run12-fixes Item 5)

---

## Existing integration architecture (the parts new sources plug into)

```
SOURCE
  ↓
[Connector — one of: git, huggingface, zenodo, etherscan, manual, audit_report, rekt_scraper]
  ↓
data/raw/<source>/repo/
  ↓
[Preprocessing — solc compile, hash, sidecar]
  ↓
data/preprocessed/<source>/<sha256>.sol + .meta.json
  ↓
[Representation — graph extraction, GraphCodeBERT tokenization]
  ↓
data/representations/<source>/<sha256>/graph.pt + tokens.pt
  ↓
[Labeling — Parser reads meta.json, Crosswalk maps source-label → SENTINEL 10-class]
  ↓
data/labels/<source>/<sha256>.labels.json
  ↓
[Merger — combines sources by confidence tier precedence (T0 > T1 > T2 > T3)]
  ↓
data/labels/merged/<sha256>.labels.json
  ↓
[Verification — pattern checkers, semantic_checker, gate (VERIFIED/PROVISIONAL/BEST-EFFORT)]
  ↓
[Splitting — stratified, dedup_enforcer, leakage_auditor]
  ↓
data/splits/v{N}/{train,val,test}.jsonl
  ↓
[Export — chunked PyG shards + parquet metadata + manifest.json]
  ↓
data/exports/sentinel-v{N}-<name>-<date>/
  ↓
SENTINEL-ml (consumes via SentinelDataset)
```

### What we already have (no new code needed)

| Component | Files | Status |
|---|---|---|
| `git` connector | `ingestion/connectors/git_connector.py` | ✅ Works for CGT (gsalzer/cgt) |
| `huggingface` connector | `ingestion/connectors/huggingface_connector.py` | ✅ Works for HF datasets (msc-auditing, jhsu12) |
| `manual` connector | `ingestion/connectors/manual_connector.py` | ✅ Works for any local-staged source |
| `etherscan` connector | `ingestion/connectors/etherscan_connector.py` | ✅ Exists (deferred) |
| `zenodo` connector | `ingestion/connectors/zenodo_connector.py` | ✅ Exists (deferred) |
| `audit_report` connector | `ingestion/connectors/__init__.py:16` | ✅ Maps to ManualConnector |
| `rekt_scraper` connector | `ingestion/connectors/__init__.py:17` | ✅ Maps to ManualConnector |
| `dedup_enforcer` | `splitting/dedup_enforcer.py` | ✅ Works (v2 file, needs regen for v4) |
| `leakage_auditor` | `splitting/leakage_auditor.py` | ✅ Works |
| `merger.py` | `labeling/merger.py` | ⚠️ Needs source added to list |
| `verification/gate.py` | `verification/gate.py` | ✅ Works (T0→VERIFIED, T2→PROVISIONAL) |
| `SentinelDataset` | `ml/src/datasets/sentinel_dataset.py` | ✅ v2/v3-compatible |

### What we need to add per new source

| Component | Required for new source? | Notes |
|---|---|---|
| **New crosswalk YAML** | YES | `<source>.yaml` in `labeling/crosswalks/` |
| **New parser Python** | YES (if source format differs) | `labeling/parsers/<source>.py` |
| **Add to `merger.py` source list** | YES | One-line addition |
| **Add to `config.yaml` source list** | YES | One source entry |
| **Add to `verification/patterns/`** | OPTIONAL | Only if existing patterns don't cover |
| **Re-preprocess / re-represent** | YES | Required for new contracts |
| **Re-label / re-merge** | YES | Required for new contracts |
| **Re-split (`--version 4`)** | YES | Don't overwrite v3 |
| **Re-export (`v4-<source>-<date>`)** | YES | Don't overwrite v3 |
| **Run all 586 data_module tests** | YES | Verify no regressions |
| **Run all 38 ml tests** | YES | Verify schema compatibility |
| **Re-run 7 readiness gates** | YES | On v4 export |

---

## Risk inventory — what can break when adding new sources

These are the 10 specific failure modes I see, ranked by severity × likelihood:

### 🔴 R1: NUM_CLASSES change (highest severity)

**Risk:** New source has a vulnerability type that doesn't map to our 10 classes (e.g., CGT might have "suicidal" or "magic_number"). If we expand `taxonomy.yaml` to add the class, `NUM_CLASSES` goes from 10 to 11+. The classifier head in `trainer.py` is hard-coded to 10 outputs. **All trained checkpoints become unusable.**

**Mitigation:**
- HARD RULE: never add a new class. Always map to one of the 10 existing classes or drop the contract.
- Crosswalk YAML MUST cover 100% of source labels (use `fallback_label: NonVulnerable` for unmappable labels, with a log warning).
- Reject any crosswalk that would require schema/taxonomy change at code review.

### 🔴 R2: v3 export gets overwritten

**Risk:** Re-running `sentinel-data export` without explicit version flag overwrites v3. Run 12 trains on v3, which gets corrupted mid-training.

**Mitigation:**
- HARD RULE: every new export uses `--version 4` minimum, with `<source>-<date>` suffix.
- v3 directory `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/` is read-only during the v4 work.
- Verify by `chmod -R a-w` on v3 dirs OR by writing to a clearly different path.

### 🔴 R3: DoS label noise re-injected

**Risk:** The DoS/Reentrancy co-occurrence patch (2026-06-13) zeroed 2,655 DoS labels. New sources (especially Slither-labeled like CGT, HF audit-firms) may have similar co-occurrence noise. If we just merge without re-applying the patch, DoS quality degrades.

**Mitigation:**
- The DoS patch lives in the merger (post-merge step). Verify the patch is re-applied at the merger for v4.
- Add a verification: `n_dos_labels_v4` should be approximately `n_dos_labels_v3 + n_dos_labels_new_source - co_occurrences_zeroed`.
- Spot-check 10 DoS=1 train contracts in v4 to confirm admin-DoS pattern is still the dominant pattern.

### 🔴 R4: Graph-hash L3 dedup file gets stale

**Risk:** `data_module/data/dedup_groups_graph_hash.json` (12,577 groups) is keyed on existing 21,523 contracts. New contracts aren't in it. The dedup_enforcer will see new contracts as `dedup_group=None` and skip them, just like the original 45% leakage bug.

**Mitigation:**
- After preprocessing + representation of new source, regenerate `dedup_groups_graph_hash.json` to include the new contracts.
- OR: write a new dedup_groups file for v4 that includes the new contracts.
- Re-verify 0% leakage on v4 splits before declaring ready.

### 🔴 R5: 36-issue regression test breaks

**Risk:** `data_module/tests/test_representation/test_13_issue_preservation.py` has 36 hand-picked contracts from the pre-Run-8 audit. Adding new sources shouldn't change these, but the test runs Slither which has version-pinned behavior. If new contracts cause Slither to behave differently, the test could fail.

**Mitigation:**
- The 36 test contracts are FROZEN (their sha256 hashes are in the test file).
- New source contracts don't go into this test set.
- If the test fails on v4 but passed on v3, the bug is in our preprocessing, not the test.

### 🔴 R6: Verification gate downgrades existing classes

**Risk:** The verification gate (`verification/gate.py`) marks classes as VERIFIED/PROVISIONAL/BEST-EFFORT based on confidence tier. If new source has a different tier for an existing class, gate output changes. E.g., if CGT (T0) has DoS labels but DIVE DoS is T2/BEST-EFFORT, the merger may upgrade DoS to VERIFIED. This changes the per-class confidence for existing v3 contracts.

**Mitigation:**
- Merger applies tier precedence: a contract with conflicting labels takes the HIGHEST tier. This is correct behavior.
- BUT: log every contract where tier changes due to v4 merger.
- Verify the per-class gate output of v4 is consistent with v3 (where no changes were expected).

### 🔴 R7: Schema version mismatch

**Risk:** `graph_schema_version: v9` is locked in manifest. If a new source's contracts trigger a different graph extraction path (e.g., a new pragma version, a new Solidity feature), Slither might emit different IR. The v9 schema assumes 12 features, 14 node types, 12 edge types. If Slither's IR for a new contract uses a node type ID > 13, we get out-of-bounds tensor access.

**Mitigation:**
- The schema is v9, but `_MAX_TYPE_ID = 13.0` — any new type ID > 13 is a hard error in representation step.
- Run the pre-Run-8 audit regression test on the new source's representations before merging.
- If any graph has type ID > 13, the source is rejected and added to a "schema-incompatible" list.

### 🟡 R8: Class distribution shift

**Risk:** Adding 3,103 CGT contracts (mostly T0 verified) shifts class distributions. If CGT is heavy on Reentrancy (it's a known reentrancy benchmark), the Reentrancy class becomes even more dominant. Macro F1 averaging gets harder.

**Mitigation:**
- v3 baseline macro F1 = 0.3293 (Run 11 ep1). If v4 macro F1 drops because of distribution shift, that's signal that the data is too lopsided.
- Nonvuln cap (currently 3 per source per split) prevents overfitting to negative class.
- Stratified splitting per source already handles within-source balance; cross-source balance is the merger's job.

### 🟡 R9: 7 readiness gates flip to RED

**Risk:** The 7 gates were GREEN on v2/v3. v4 might trigger new gate failures:
- Gate 5 (all 10 classes verified): CGT might not have all 10 classes → some classes drop to corpus-bound YELLOW
- Gate 6 (no leakage): if dedup_groups file is stale, leakage comes back
- Gate 7 (no open code-bug regression): if CGT triggers the BUG-H7 EMITS edge issue or the predictor F8/F10 fix breaks

**Mitigation:**
- All 7 gates re-run on v4. Any RED is a blocker.
- YELLOW is acceptable if corpus-bound (e.g., a class with 0 positives in CGT).
- Document the gate results in the v4 plan section.

### 🟢 R10: Export format incompatibility

**Risk:** The v3 export is in `format_schema/v1` with 5+5 shards, parquet metadata, hash-verified manifest. A new source might add contracts that push the shard count from 5 to 6+. The SentinelDataset's `@lru_cache(maxsize=4)` shard cache would need updating.

**Mitigation:**
- `SENTINEL_SHARD_CACHE_SIZE` env var already exists for this.
- Default cache size of 4 is fine for ≤5 shards; bump to 6 if v4 has 6 shards.
- The SentinelDataset's 3 hard `ValueError` gates at `__init__` will catch any format mismatch.

---

## Risk mitigation framework — the protocol

For EVERY new source, follow this 5-stage protocol. No skipping.

### Stage 1: Sandbox (DO NOT touch v3)

```bash
# Create a worktree or branch
git worktree add ../sentinel-cgt -b feat/cgt-ingest

# Stage the new source
mkdir -p data_module/data/raw_staging/cgt/
# ... copy data ...

# Create crosswalk + parser (separate files from v3)
touch data_module/sentinel_data/labeling/crosswalks/cgt.yaml
touch data_module/sentinel_data/labeling/parsers/cgt.py
```

**Done means:** Files exist, but no code in `merger.py` or `config.yaml` references the new source.

### Stage 2: Ingest + preprocess (source-only)

```bash
# Use the new source
sentinel-data ingest --source cgt --connector git
sentinel-data preprocess --source cgt
sentinel-data represent --source cgt
```

**Done means:** New source has `data_module/data/preprocessed/cgt/*.sol` and `data_module/data/representations/cgt/*/graph.pt`. **v3 export is untouched.**

### Stage 3: Label + crosswalk test

```bash
sentinel-data label --source cgt
```

**Done means:** 
- `data_module/data/labels/cgt/*.labels.json` files written
- Every label maps to one of the 10 SENTINEL classes (no "unknown" class emitted)
- All 586 data_module tests still pass
- v3 export's `labels.parquet` is unchanged

### Stage 4: Merge + split (versioned, no overwrite)

```bash
# Add CGT to merger.py source list
# Run merger to produce merged labels
sentinel-data merge --include cgt --output data_module/data/labels/merged/

# Re-run dedup with new contracts
sentinel-data dedup --version 4

# Re-split (DO NOT overwrite v3)
sentinel-data split --version 4 --output data_module/data/splits/v4/
```

**Done means:**
- `data_module/data/splits/v4/{train,val,test}.jsonl` exists
- v3 splits (`v3/`) are byte-identical to before
- 0% cross-split leakage on v4 verified

### Stage 5: Export v4 + gates + tests

```bash
# Export to v4 path (NEVER to v3 path)
sentinel-data export --output data_module/data/exports/sentinel-v4-cgt-2026-06-XX/

# Run all 7 readiness gates on v4
sentinel-data gate-check --export-dir data_module/data/exports/sentinel-v4-cgt-2026-06-XX/

# Run all tests
pytest data_module/tests/ -v
pytest ml/tests/ -v
```

**Done means:**
- v4 export exists with manifest hash verified
- 6/7 gates GREEN (1 acceptable YELLOW for corpus-bound class is OK)
- All 586 data_module + 38 ml tests pass

---

## Source-by-source plan

### Source #1: CGT (gsalzer/cgt) — RECOMMENDED FIRST

| Field | Value |
|---|---|
| **Source** | `gsalzer/cgt` on GitHub |
| **Volume** | 3,103 source contracts / 2,529 deployment / 2,473 runtime |
| **Labels** | 20,455 manually checked assessments |
| **Connector** | `git` (already implemented) |
| **Tier** | T0 (manual verification by academic team) |
| **Crosswalk** | NEW: `cgt.yaml` |
| **Parser** | NEW: `cgt.py` (mirror `smartbugs_curated.py` shape) |
| **Effort** | 1-2 days |
| **Expected F1 lift** | +0.02-0.05 on classes that map well (Reentrancy, IntegerUO, Timestamp, AccessControl) |
| **Expected risk** | LOW (T0 labels, manual verification) |

**Label mapping challenge:** CGT uses 11 vulnerability categories per Salzer's paper. Need to map:
- `reentrancy-eth` → Reentrancy
- `reentrancy-no-eth` → Reentrancy
- `arithmetic` → IntegerUO
- `suicidal` → ExternalBug (closest)
- `TOD` → TransactionOrderDependence
- `tx-origin` → ExternalBug
- `magic_number` → ??? (no clean mapping, drop with log warning)
- `access_control` → ExternalBug
- `unchecked_low_level` → UnusedReturn
- `time_manipulation` → Timestamp
- `denial_of_service` → DenialOfService

**Class gains (estimated):**
- Reentrancy: 7,950 → 8,500+ (~+550)
- ExternalBug: 16,621 → 17,000+ (some suicidal/tx-origin)
- UnusedReturn: 3,500 → 3,700+ (~+200)
- IntegerUO: 3,500 → 3,800+ (~+300)
- Timestamp: 1,500 → 1,700+ (~+200)
- TransactionOrderDependence: 643 → 700+ (~+50-100)

**Pipeline integration steps:**
1. `git clone https://github.com/gsalzer/cgt` → `data_module/data/raw_staging/cgt/`
2. Create `cgt.yaml` crosswalk with the mapping above + `fallback_label: NonVulnerable`
3. Create `cgt.py` parser (mirror `smartbugs_curated.py` — one-level folder structure)
4. Add CGT to `merger.py` source list
5. Add CGT to `config.yaml`
6. Run 5-stage protocol above
7. Verify: 0% leakage, all 7 gates pass (Gate 5 may flip to YELLOW if CGT has no DoS labels)

### Source #2: HF audit-firms (`msc-smart-contract-auditing/vulnerability-severity-classification`) — DEFERRED

| Field | Value |
|---|---|
| **Source** | HF dataset, 2,910 functions |
| **Sources** | Codehawks, ConsenSys, Cyfrin, Sherlock, Trust Security |
| **Labels** | `none / low / medium / high` severity |
| **Connector** | `huggingface` (already implemented) |
| **Tier** | T1 (real audit firm findings) |
| **Crosswalk** | NEW: `hfaudit.yaml` |
| **Parser** | NEW: `hfaudit.py` (different from solidifi.py — function-level, severity-to-class mapping) |
| **Effort** | 3-5 days (severity mapping is non-trivial) |
| **Expected F1 lift** | Unknown — depends on severity-to-class mapping quality |

**Why deferred:** The severity-to-class mapping is genuinely hard. `medium` could mean IntegerUO or DoS depending on context. Need a per-pattern classification rule, not just severity.

**Recommended deferral criteria:** Re-evaluate after Source #1 (CGT) is shipped, with a clear methodology for severity mapping.

### Source #3: Kaggle synthetic (`jhsu12/smart_contract_vulnerability_kaggle`) — DEFERRED

| Field | Value |
|---|---|
| **Source** | HF dataset, 10,436 rows |
| **Labels** | vulnerability_type, severity, root_cause, vulnerable_code, fixed_code, mitigation |
| **Connector** | `huggingface` (already implemented) |
| **Tier** | T3 (synthetic patterns, no real exploitation) |
| **Crosswalk** | NEW: `kaggle_synth.yaml` |
| **Parser** | NEW: `kaggle_synth.py` (parses JSON, maps vulnerability_type to SENTINEL class) |
| **Effort** | 1-2 days (if diversity check passes) |
| **Expected F1 lift** | +0.005-0.02 (synthetic, low signal) |

**Why deferred:** Need to verify what vulnerability types are covered. The preview shows 100% Reentrancy (useless since we have 7,950 Reentrancy train). Need to download the full dataset and check the distribution.

**Recommended deferral criteria:** Diversity check first. If ≥6 of our 10 classes are represented, proceed. If <4, skip.

### Sources we should NOT add (decision logged here)

| Source | Why skip |
|---|---|
| `biagioboi/ETH_SmartContractVulnerability_LLM` | LLM-generated, no real labels, 17 downloads |
| Messi-Q Resource 1 (40K unlabeled) | Needs Slither relabeling pass, days of work, weak ROI |
| Skelcodes (248K) | Just an Etherscan availability indicator, no labels |
| DeFiHackLabs | BLOCKED in pre-run12-fixes (PoC contracts, not vulnerable contracts) |

---

## Order of operations

### Phase 1: Run 12 unblocked (the pre-Run-12 plan stays the priority)

1. **Do Step B + Step D from pre-run12-fixes-2026-06-13.md** (Level-3 dedup + 7 gates on v3)
2. **Launch Run 12 on v3 export** — Run 11 ep1 is the baseline (F1=0.3293)
3. **Document Run 12 results**

**Why first:** Run 12 produces portfolio evidence in 1-2 days. CGT ingestion is 1-2 weeks. The order is portfolio-impact-driven.

### Phase 2: CGT ingestion (the highest-value new source)

1. **Sandbox CGT** (Stage 1 of protocol)
2. **Ingest + preprocess CGT** (Stage 2)
3. **Crosswalk + label CGT** (Stage 3)
4. **Merge + split v4** (Stage 4)
5. **Export v4 + gates + tests** (Stage 5)
6. **Document v4 readiness** (parallel to pre-run12-fixes Step D template)

### Phase 3: Defer HF audit-firms and Kaggle until Run 12 results analyzed

The decision to add Source #2 or #3 should be data-driven, not optimistic. If Run 12 on v3 hits F1=0.40+, the marginal value of CGT is small (we're already at portfolio-acceptable). If Run 12 is still 0.33-0.35, CGT is essential.

---

## Rollback strategy

If anything in Phase 1 or Phase 2 breaks:

### v3 export is the rollback target

- v3 splits: `data_module/data/splits/v3/` — never modified
- v3 export: `data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/` — never modified
- v3 dedup file: `data_module/data/dedup_groups_graph_hash.json` — backed up before v4 work
- v3 labels: `data_module/data/labels/merged/` — backed up before v4 work
- v3 crosswalks: all 4 crosswalk YAMLs — backed up to `data_module/temp/crosswalks_backup_v3/`

### How to roll back

```bash
# If v4 ingestion failed:
rm -rf data_module/data/preprocessed/cgt/
rm -rf data_module/data/representations/cgt/
rm -rf data_module/data/labels/cgt/
rm -rf data_module/data/splits/v4/
rm -rf data_module/data/exports/sentinel-v4-*/

# v3 still trains Run 12 with zero impact
```

### What we can NEVER roll back

- Schema version changes (v9 → v10) — once bumped, never revert
- NUM_CLASSES changes (10 → 11+) — once bumped, all checkpoints invalid
- Taxonomy.yaml class additions — same as NUM_CLASSES

This is why R1 (NUM_CLASSES) and R7 (schema version) are the highest-severity risks.

---

## Testing strategy

### Unit tests (per new source)

| Test | Purpose | File |
|---|---|---|
| Crosswalk recall test | Every source label maps to a SENTINEL class | `data_module/tests/test_labeling/test_<source>_recall.py` |
| Parser test | Parser handles all known folder/file layouts | `data_module/tests/test_labeling/test_<source>_parser.py` |
| Schema compatibility test | No graph has type ID > 13 | `data_module/tests/test_representation/test_schema_compat.py` |
| Tier assignment test | Every contract has a valid confidence tier | `data_module/tests/test_verification/test_tier_<source>.py` |

### Integration tests (per v4 export)

| Test | Purpose |
|---|---|
| Leakage audit | 0% cross-split leakage on v4 splits |
| Gate 1-7 re-run | All 7 v2-readiness gates on v4 export |
| Byte-identical regression | SentinelDataset loads v4 without errors |
| 36-issue pre-Run-8 audit | Still passes (no new issues introduced) |
| `test_13_issue_preservation.py` | 36 hand-picked contracts still extract correctly |

### End-to-end test (per v4 export)

| Test | Purpose |
|---|---|
| `sentinel_dataset.py` smoke test | 15,000 train / 3,000 val loads in <30s |
| `trainer.py` 1-epoch smoke test | Train on 1,000-sample subset for 1 epoch without errors |
| Per-class F1 (1 epoch) | Compare to v3 ep1 baseline |

---

## Cross-references

### Pre-existing related plans

- `data_module/temp/pre-run12-fixes-2026-06-13.md` — Run 12 readiness, Items 1-8
- `data_module/docs/architecture.md` — overall v2 data module architecture
- `data_module/docs/v2-readiness-2026-06-12.md` — the 7 readiness gates
- `data_module/audit/v2_full_audit/06_FINAL_master_report.md` — prior audit
- `data_module/DATA_MODULE_AUDIT.md` — AI's audit (45% leakage finding)

### Memory

- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — current state
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run11_launch.md` — Run 11 specifics
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_stage7b_handoff.md` — seam swap story

### External references

- CGT (gsalzer/cgt) — https://github.com/gsalzer/cgt
- HF audit-firms — https://huggingface.co/datasets/msc-smart-contract-auditing/vulnerability-severity-classification
- HF Kaggle synthetic — https://huggingface.co/datasets/jhsu12/smart_contract_vulnerability_kaggle
- SmartBugs Curated (already ingested) — https://github.com/smartbugs/smartbugs-curated

---

## Open questions (need Ali's decision)

1. **CGT branch strategy:** Should CGT ingestion happen on a feature branch (`feat/cgt-ingest`) or directly on main? Recommended: feature branch, merge to main only after v4 export passes all gates.

2. **v4 export naming:** Should we name it `sentinel-v4-cgt-2026-06-XX` (per source) or `sentinel-v4-merged-2026-06-XX` (per export)? Recommended: per-source, so we can have v4-cgt, v4-cgt-kaggle, etc.

3. **CGT class gains priority:** Do we want to add CGT even if the gain is mostly on already-strong classes (Reentrancy +550), or skip and find a source that helps dead classes (CallToUnknown, GasException, MishandledException)?

4. **CGT vs DeFiHackLabs:** CGT is recommended first because it's T0/T1 quality academic data. DeFiHackLabs is BLOCKED in pre-run12-fixes. If we unblock DeFiHackLabs (extract vulnerable protocol contracts, not PoC), it might be higher-value than CGT. Re-evaluate after Run 12 results.

5. **HF audit-firms scope:** Even with severity mapping complexity, the 5 audit firms (Codehawks, ConsenSys, Cyfrin, Sherlock, Trust Security) are world-class. Should we hire/freelance the severity-to-class mapping work, or skip and find cleaner T0 sources?

6. **Synthetic data policy:** Should SENTINEL accept synthetic data (jhsu12 Kaggle) as T3-bronze, or hold to "real exploitation only"? The latter is purer but limits us to maybe 25-30K contracts total.

---

## Final summary

| Phase | Action | Effort | Output |
|---|---|---|---|
| **Phase 1 (NOW)** | Complete pre-run12-fixes Steps B + D; launch Run 12 | 1-2 days | Run 12 baseline F1 |
| **Phase 2 (post-Run 12)** | CGT ingestion per 5-stage protocol | 1-2 weeks | v4 export, +3,103 contracts, +20K labels |
| **Phase 3 (deferred)** | HF audit-firms + Kaggle synthetic | 1-2 weeks each | v5+ exports |
| **Phase 4 (separate track)** | DeFiHackLabs extraction (if unblocked) | 2+ weeks | v? export with real exploits |

---

## 🆕 POST-RUN-12 UPDATE (2026-06-14)

**BCCC re-evaluation found ONLY MishandledException is extractable from BCCC.** 658 high-confidence contracts (547 both-tool + 85 slither-only + 26 verified aderyn-only) ready for Run 13 injection. **Other 4 classes (GasException, CallToUnknown, DoS, ToD) are noise in BCCC** (0-25% 2-tool consensus + 40-92% compile failure). See `~/.claude/projects/.../memory/project_bccc_2tool_audit_2026-06-14.md` for full audit.

**Updated Phase 2 plan (post-Run-12):**
1. **BCCC ME injection (NEW PRIORITY)** — 658 contracts, 1-2 weeks, adds 13.5x to v3 ME
2. CGT ingestion (per 5-stage protocol) — original Phase 2 plan
3. DeFiHackLabs extraction (if unblocked) — original Phase 4
4. Drop GasException (NUM_CLASSES=9) — see feature_leakage_audit_2026-06-14.md

**Skip:**
- HF audit-firms for MishandledException (BCCC covers this gap)
- Kaggle synthetic for MishandledException (BCCC is real data, not synthetic)

**HARD RULES (apply forever):**
1. Never change NUM_CLASSES, schema version, or taxonomy.yaml without explicit Ali sign-off
2. Never overwrite v3 export, v3 splits, or v3 dedup file
3. Always version up: v4, v5, v6 — never replace
4. Every new source follows the 5-stage protocol
5. Every new export passes all 7 readiness gates
6. Every new export's `trainer.py` smoke test runs for 1 epoch before declaring ready
7. Code changes to `sentinel_data` are read-only on `data/` (only `ml/` writes to checkpoints)

---

End of data source addition plan.
