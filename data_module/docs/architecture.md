# sentinel-data Architecture (v2 → v3)

> **Last revised:** 2026-06-13 23:35 UTC (post-Stage 7 + post-data-quality investigation + post-45% leakage fix + Run 11 ep1 + **DoS/Reentrancy co-occurrence patch APPLIED + Run 12 LAUNCHED**)
> **Scope:** Full v2 data module — Stages 0 through 7B complete (Stage 5 registry partially skipped). **v3 export is now ACTIVE** (post-DoS-patch).
> **Status:** End-to-end functional. **Run 12 training on v3 export** (PID 230342, launched 2026-06-13 23:31:17 UTC). 22,493 contracts / 18,596-1,983-1,914 splits / 0% leakage / 0 DoS+Reentrancy overlap / DoS=1,101 (was 3,756 pre-patch).

> **⚠️ Important:** This doc was originally written for the v2 state (Run 11). Sections below still describe v2 architecture (which is unchanged), but the **active export, splits, and DoS counts are now v3** (post-DoS-patch). See §"v2 → v3 transition" below for the change summary. Full audit trail in `data_module/temp/pre-run12-fixes-2026-06-13.md` and `data_module/temp/data-source-addition-plan-2026-06-13.md`. Run 12 launch context: `~/.claude/projects/.../memory/project_run12_launch.md` (or `project_dos_patch_2026-06-13.md` for the patch audit).

---

## 1. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        sentinel-data (v2)                          │
│                  ~/projects/sentinel/data_module/                    │
│                                                                     │
│  Sources  →  Ingest  →  Preprocess  →  Represent                    │
│                              ↓                 ↓                   │
│                          Dedup L1+L2       (graph + token shards)   │
│                                ↓                 ↓                   │
│                            Label  ←  Verify  ←  Slither caching     │
│                                ↓                                   │
│                       Split (v2 dedup)                              │
│                                ↓                                   │
│                  Export (shards + manifest)                          │
│                                ↓                                   │
│              sentinel-ml (one-way consumer)                          │
│   sentinel_dataset.py + sentinel_collate_fn + trainer.py            │
└─────────────────────────────────────────────────────────────────────┘
```

**One-way dependency:** `sentinel-ml` depends on `sentinel-data`. Never the reverse. Enforced at install time via `sentinel-data = {path = "../data_module", develop = true}` in `ml/pyproject.toml`.

---

## 2. Data flow (per-stage)

```
┌────────────────┐     ┌──────────────────┐    ┌──────────────────┐
│   Raw sources  │────▶│     Ingestion     │───▶│  Preprocessing   │
│  DIVE (22,073) │     │ Git/Zenodo/      │    │ Flatten → 2-pass  │
│  SolidiFI (283)│     │ HF/Etherscan/    │    │ compile → dedup   │
│  DeFiHackLabs  │     │ Manual connector │    │ L1 (sha256) + L2 │
│  (deferred)    │     │ SHA-256 manifest │    │ (Ethereum addr)  │
└────────────────┘     └──────────────────┘    │ (L3 STUB)         │
                                                └────────┬─────────┘
                                                         ▼
┌────────────────┐     ┌──────────────────┐    ┌──────────────────┐
│    Export      │◀────│     Splitting    │◀───│    Labeling       │
│ Sharded .pt +  │     │ stratified (70/  │    │ Per-source       │
│ parquet +      │     │ 15/15) → dedup_  │    │ parsers → merger │
│ manifest.json  │     │ enforcer →       │    │ (tier precedence)│
│ (5+5 shards,   │     │ nonvuln cap      │    │ DoS/Reentrancy   │
│ hash-verified) │     │ v2: 18,458/      │    │ patch (1,095     │
│                │     │ 1,996/1,902      │    │ pure DoS)        │
└────────────────┘     │ 0% leakage ✓     │    └────────┬─────────┘
        ▲              └──────────────────┘             │
        │                                                ▼
        │                                       ┌──────────────────┐
        │              ┌──────────────────┐    │   Verification   │
        │              │   Representation │    │ class_auditor +  │
        │              │ graph_extractor + │    │ semantic_checker │
        │              │ orchestrator +    │    │ + gate +         │
        │              │ tokenizer         │    │ patterns (10     │
        │              │ (v9 schema:       │    │ YAML)            │
        │              │ 12-dim features,  │    │ "99% DoS↔Reent   │
        │              │ 14 node types,    │    │  → patched"      │
        │              │ 12 edge types)    │    └──────────────────┘
        │              └─────────┬────────┘
        │                        │
        │              ┌─────────▼────────┐
        └──────────────│  sentinel-ml     │
                       │ SentinelDataset  │
                       │ + collate +       │
                       │ trainer.py        │
                       │ (consumes v2 exp) │
                       └──────────────────┘
```

**Active sources** (per `manifest.json`):
- **DIVE** (`source: dive`, T2) — 22,073 preprocessed contracts, 22,330 raw `.sol` files in `repo/__source__/`
- **SolidiFI** (`source: solidifi`, T0) — 283 contracts with injected vulnerabilities (highest-confidence tier)
- **DeFiHackLabs** — preprocessed but labeling deferred (forge-std import failures)

---

## 3. Per-stage contracts

| Stage | Input | Output | Schema | Verified numbers (2026-06-13) |
|---|---|---|---|---|
| **ingest** | `config.yaml` enabled sources | `data/raw/<source>/*.sol` + `ingestion_manifest.json` | — | 2 active sources, 22,613 raw .sol files |
| **preprocess** | `data/raw/` `.sol` | `data/preprocessed/<source>/<sha256>.sol` + `.meta.json` | sidecar v1 | **v3: 22,493 contracts**, Level 1+2 dedup + L3 text-hash (83/147 groups applied) |
| **represent** | `data/preprocessed/` | `data/representations/<source>/<sha256>/graph.pt` + `tokens.pt` | **graph v9** (12 features, 14 node types, 12 edge types) | **v3: 21,657 representations**; uses graphcodebert-base [4,512] |
| **label** | `data/representations/` + 4 crosswalk YAMLs (solidifi, dive, smartbugs_curated, defihacklabs-deferred) | `data/labels/merged/<sha256>.labels.json` | **class order LABELING** (per ADR-0009) | **v3: 22,493 contracts**; DoS patch applied (**1,101 pure DoS**, 0 DoS+Reentrancy overlap) |
| **verify** | `data/labels/merged/` | `data/verification/verification_report.md` | — | 91/91 tests pass; 10 patterns YAML; 1 VER + 3 PROV + 2 BEST-EFFORT classes |
| **split** | `data/labels/merged/` + `data/dedup_groups_graph_hash.json` | **`data/splits/v3/{train,val,test}.jsonl`** | — | **v3: 18,596/1,983/1,914, 0% leakage verified** (graph-hash + L3 text-hash dedup) |
| **registry** | — | `data/registry/` (catalog) | — | **SKIPPED** (Stage 5b; `data/registry/` empty placeholder) |
| **analyze** | `data/labels/merged/` + `data/preprocessed/` | `data/analysis/<run_id>/complexity_proxy_risk.md` | — | Run on v2 baseline: HIGH-RISK pairs=0 (GREEN canary) |
| **export** | **`data/splits/v3/`** + `data/representations/` | **`data/exports/sentinel-v3-smartbugs-2026-06-13/{graphs,tokens,labels,metadata,manifest}`** | **format_schema v1** | **v3: 22,493 contracts, 21,657 with reps, 5+5 shards, hash verified** (artifact_hash `5cc5cfcbf42bef4ced58b963ef98241bcf3ec4ab3bea5d198f336ec763a4faa9`) |

---

## 4. The v9 schema (the canonical contract)

```python
# data_module/sentinel_data/representation/graph_schema.py (source of truth)
FEATURE_SCHEMA_VERSION = "v9"
NODE_FEATURE_DIM = 12          # 11 base + 1 in_unchecked_block
NUM_NODE_TYPES  = 14          # 13 base + 1 CFG_NODE_ARITH
NUM_EDGE_TYPES  = 12          # 11 base + 1 EXTERNAL_CALL
_MAX_TYPE_ID    = 13.0        # v9: 14 types, IDs 0-13

CLASS_NAMES = [  # LABELING order (per ADR-0009, replaces the old representation order)
    "CallToUnknown",              # 0
    "DenialOfService",            # 1  (post-patch: 1,095 pure positives)
    "ExternalBug",                # 2  (dominates: 16,621 positives, 77% prevalence)
    "GasException",               # 3  (dead: 0 positives — needs SmartBugs)
    "IntegerUO",                  # 4
    "MishandledException",        # 5  (39 positives — corpus-bound)
    "Reentrancy",                 # 6
    "Timestamp",                  # 7
    "TransactionOrderDependence", # 8
    "UnusedReturn",               # 9
]
NUM_CLASSES = 10
```

**Seam swap architecture** (the critical Stage 7B change):
- `sentinel_data.representation.graph_schema` = source of truth (full v9 schema, self-contained)
- `ml/src/preprocessing/graph_schema.py` = 18-line thin re-export shim
- `ml/src/preprocessing/graph_extractor.py` lines 112 + 1241 = import directly from `sentinel_data.representation.graph_schema` (avoids circular import via the shim)

---

## 5. The 45% leakage fix (the most important thing in this doc)

**Discovery:** AI assistant's full audit (2026-06-13) found that v1 splits had 10,811 / 21,523 contracts (50.2%) as **exact graph duplicates** (same `x + edge_index` bytes), with **45% of val and 45% of test contracts having an identical graph in train**. Run 10 reached F1=0.683 at ep32 — **that was memorization, not learning.**

**Root cause:** `_run_split` in `data_module/sentinel_data/cli.py` built `Contract` objects from `labels/merged/*.labels.json` without setting `dedup_group`. The `dedup_enforcer` then skipped all contracts where `c.dedup_group is None` → `dedup_groups_resolved: 0` → no dedup enforcement.

**3-part fix:**
1. **Graph content hash dedup file** — `data_module/data/dedup_groups_graph_hash.json` (12,577 unique groups from 21,523 contracts; MD5 of `x.tobytes() + edge_index.tobytes()` per contract)
2. **`cli.py:_run_split` CLI fix** — reads the dedup file, passes `dedup_group=cid_to_group.get(sha)` to each `Contract`
3. **Re-ran `sentinel-data split --version 2`** — 18,458/1,996/1,902 with **0% leakage verified** at both dedup_group and graph-tensor levels

**Proper long-term fix (still pending):** Implement Level-3 dedup in `data_module/sentinel_data/preprocessing/deduplicator.py:73` (currently a STUB). The current workaround addresses the worst case but isn't architecturally clean.

---

## 6. The v2 export format

`data_module/data/exports/sentinel-v2-baseline-2026-06-12/`:
```
manifest.json                 ← written LAST (contains artifact_hash)
labels.parquet               ← all 22,356 labeled contracts (all splits)
metadata.parquet             ← same coverage, enriched from sidecars
graphs/
  graphs-00000.pt            ← PyG Batch, up to 5,000 contracts
  graphs-00001.pt            ← 5 shards × ~5,000 contracts = 21,523 total
  graphs-00002.pt
  graphs-00003.pt
  graphs-00004.pt            ← 1,523 contracts (final shard)
tokens/
  tokens-00000.pt            ← torch.Tensor [N, 4, 512] int64
  tokens-00001.pt            ← 5 shards, same ordering as graphs/
  tokens-00002.pt
  tokens-00003.pt
  tokens-00004.pt
```

**Manifest fields (key ones):**
- `schema_version: "v1"` (format contract version)
- `graph_schema_version: "v9"` (matches the v9 schema)
- `n_contracts: 22356` · `n_contracts_with_reps: 21523`
- `shard_index: {sha256: {shard: int, pos_in_shard: int}}` — used by `SentinelDataset` to locate each contract
- `splits: {train: [sha256, ...], val: [...], test: [...]}` — 18,458/1,996/1,902
- `artifact_hash: <md5>` — covers all files EXCEPT `manifest.json` itself (avoids chicken-and-egg)

**`SentinelDataset` 5-tuple return:**
```python
(graph: Data, tokens: dict, y: Tensor[10], contract_id: str, confidence_tier: str | None)
```
Where `tokens = {"input_ids": [4,512], "attention_mask": [4,512]}` (attention_mask reconstructed as `(input_ids != 1).long()` because graphcodebert-base's pad_token_id=1).

**3 hard `ValueError` gates at `__init__`:**
1. `manifest.schema_version == "v1"` (forward-compat)
2. `manifest.graph_schema_version == "v9"` (matches model expectations)
3. `verify_artifact_hash()` returns True (data-integrity check)

---

## 7. The 7 v2-readiness gates (current state)

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Schema regression (Stage 2 byte-identical) | ✅ GREEN | 40/40 tests pass (after fixing 2 latent bugs) |
| 2 | BCCC Phase 5 verification regression | ✅ GREEN | 191/212 tests pass (21 skip on solc/external) |
| 3 | End-to-end round-trip (SentinelDataset) | ✅ GREEN | 16/16 unit + smoke test (15,063 train + 3,226 val) |
| 4 | Feature distribution (Stage 6 canary) | ✅ GREEN | `high_risk_pairs=0` in v2 baseline; pos>neg complexity bias hidden under "GREEN" |
| 5 | All 10 classes VERIFIED or PROVISIONAL | 🟡 YELLOW | 1 VER + 3 PROV + 2 BEST-EFFORT (corpus-bound); 3 dead/tiny classes (GasException=0, CallToUnknown=39, MishandledException=39) |
| 6 | No leakage across splits | ✅ GREEN | 0% on v2 deduped splits (was 45% on v1) |
| 7 | No open code-bug regression | ✅ GREEN | EMITS 4/4 + predictor per-class thresholds confirmed in code |

**Overall: 6 GREEN, 1 acceptable YELLOW, 0 RED.** Run 11 is the first honest measurement on clean splits.

---

## 8. Run 11 epoch 1 (the first honest F1)

| Metric | Value | Notes |
|---|---|---|
| **f1_macro (fixed)** | **0.3293** | First honest F1 on v2 deduped splits (vs Run 7's 0.3423, Run 9's 0.3081) |
| **f1_macro_tuned** | **0.3385** | +0.009 from per-class threshold tuning |
| Epoch duration | 25.7 min | Faster than projected (I estimated 3.4 hr/ep — actual is 25.7 min) |
| Checkpoint | `ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt` (280 MB) | SAVED, can be resumed |
| Status | 🟢 IN TRAINING (was PID 556540; WSL crashed during epoch 2) | Need to resume |

**Per-class F1 at epoch 1** (from `epoch_summary.jsonl`):
| Class | F1 | AUC-PR | Interpretation |
|---|---|---|---|
| ExternalBug | 0.862 | 0.818 | Dominates (77% prevalence); model at base-rate performance |
| Reentrancy | 0.648 | 0.655 | Solid signal |
| IntegerUO | 0.614 | 0.572 | Solid |
| Timestamp | 0.504 | 0.395 | OK |
| UnusedReturn | 0.452 | 0.419 | OK |
| TransactionOrderDependence | 0.109 | 0.086 | Bad — small (643 positives) |
| **DenialOfService** | **0.104** | **0.046** | **Near-random** — reviewer's flag is VERIFIED |
| CallToUnknown | 0.000 | 0.006 | Dead (39 positives) |
| GasException | 0.000 | NaN | Dead (0 positives) |
| MishandledException | 0.000 | 0.003 | Dead (39 positives) |

**Macro F1 is inflated by ExternalBug** (at base rate). Honest signal is in per-class breakdown.

---

## 9. One-way dependency boundary

```
sentinel-ml (ml/pyproject.toml)
  └── depends on: sentinel-data = {path = "../data_module", develop = true}

sentinel-data (data_module/pyproject.toml)
  └── NO dependency on sentinel-ml
```

**Enforced at install time** via `poetry install` in `ml/`. The sentinel-ml module imports from `sentinel_data` (with a thin shim layer in `ml/src/preprocessing/`).

**After Stage 7B:** the seam swap means `sentinel-ml` doesn't need to import from the shim — it can import directly from `sentinel_data.representation.graph_schema` (which is the source of truth) for any new code.

---

## 10. The 6 critical tests (structural defense)

1. **36-issue pre-Run-8 audit regression** (`data_module/tests/test_representation/test_13_issue_preservation.py`) — every A1–A38 fix preserved through port
2. **Byte-identical regression** (`data_module/tests/test_representation/test_byte_identical_regression.py`) — new path output = old `ml/` path output (also fixed 2 latent bugs in this session)
3. **BCCC Phase 5 regression** (`data_module/tests/test_verification/test_bccc_regression.py`) — new verification matches Phase 1-5 results ±0.5%
4. **EMITS edge fixture test** (`data_module/tests/test_representation/test_emits_fixture.py`) — 4 tests verify the BUG-H7 fix
5. **7 v2-readiness gates** — see §7 above
6. **Graph-hash dedup verification** (my check in this session) — 0% cross-split leakage verified independently

**Test counts (2026-06-13):**
- `data_module/tests/`: 586/613 pass (27 skip on solc/external)
- `ml/tests/`: 38/38 ml tests pass (test_trainer + test_sentinel_dataset + test_predictor)

---

## 11. Known open items (deferred beyond Stage 7)

| Item | Severity | Owner | Notes |
|---|---|---|---|
| **Level-3 dedup** (deduplicator.py:73 stub) | 🟡 P2 | v2.1 | Proper fix; current workaround is the graph-hash file |
| **DoS label quality** (AUC-PR=0.046 near-random) | 🟡 P2 | needs investigation | Reviewer flag is VERIFIED; investigate before another Run |
| **CALL_ENTRY cross-function** | 🟢 P3 | post-Run 11 | Currently self-loop only; full fix in v2.1 |
| **C-4 max_nodes=2048 truncation** | 🟢 P3 | post-Run 11 | A graph with 2,904 nodes was truncated; rate unknown (warning fires once) |
| **DeFiHackLabs ingestion** | 🟡 P2 | v2.1 | Preprocessed but not labeled (forge-std import failures) |
| **SmartBugs Curated ingestion** | 🟠 P1 | v2.1 | Would lift GasException=0, CallToUnknown=39, MishandledException=39 |
| **Stage 5 registry (SQLite catalog)** | 🟡 P3 | v2.1 | Skipped to reach Run 11; needed for production promotion |
| **drift basline.json** | 🟢 P4 | post-Run 11 | Replace placeholder with real warmup output |
| **Stage 5.5 GCB propagation** | 🟠 P1 | v2.1 | Improve DoS/Timestamp label confidence |
| **Pre-0.8 `is_checked` fix** (`_detect_pre_08` helper in graph_extractor.py) | 🟢 P3 | v2.10 | Code ready, re-export pending. Schema bumps to v10 when applied |

---

## 12. Confidence tier system (per `verification/patterns/`)

| Tier | Description | Sources (this v2) |
|---|---|---|
| T0 | Verified exploit — on-chain proof | (none in v2 export — SolidiFI is "inject-verified" but maps to T0) |
| T1 | Gold — human-curated or mathematically certain | (none in v2 — DeFiHackLabs deferred) |
| T2 | Silver — expert auditors or 3/5 tool majority | DIVE (22,073 contracts) |
| T3 | Bronze — tool-generated, conservative threshold | (none in v2 export) |
| T4 | Unlabeled — pretraining / NonVulnerable class only | (none in v2 export) |

**The 5-tier system is in the code** (10 patterns YAML) but the v2 export's only sources are DIVE (T2) and SolidiFI (T0). SmartBugs Curated (T3) is in the v2 plan but ingestion is deferred.

---

## 13. Cross-references (where to look for what)

### Design decisions
- **ADR-0008** (`docs/decisions/ADR-0008-export-and-seam-swap-design.md`) — export + seam swap design + 7B Amendment (11 subsections)
- **ADR-0009** (`docs/decisions/ADR-0009-canonical-class-vocabulary.md`) — LABELING order is the canonical 10-class vocabulary
- `docs/proposal/Data_Module_Proposals/README.md` — top-level v2 build plan

### Data quality + leakage audit
- **`data_module/DATA_MODULE_AUDIT.md`** (489 lines) — the AI assistant's full pipeline audit; the 45% leakage finding
- **`data_module/audit/v2_full_audit/07_data_quality_investigation.md`** (394 lines) — my 6-gap investigation
- **`data_module/docs/v2-readiness-2026-06-12.md`** (12.5 KB) — the 7 v2-readiness gates report
- `data_module/audit/v2_full_audit/01-06_*.md` — the 5-phase prior audit (4 phase reports + master + plans)

### Live working plans
- `data_module/temp/live_plans/stage_7a_export_module.md` — Stage 7A implementation log
- `data_module/temp/live_plans/stage_7b_seam_swap_active.md` — Stage 7B implementation log

### Run 11
- `ml/logs/GCB-P1-Run11-v2deduped-20260613.log` — plain log
- `ml/logs/GCB-P1-Run11-v2deduped-20260613/{step_metrics,epoch_summary,alerts}.jsonl` — structured logs
- `ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt` — current best (ep1 F1=0.3293)
- `ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.state.json` — resume state

### Memory (project-level)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` — current state (230 lines)
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run11_launch.md` — Run 11 specifics
- `~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_stage7b_handoff.md` — seam swap story

---

## 14. Quick reference: file paths of the 14 most important files

```
# Source (data module)
data_module/sentinel_data/representation/graph_schema.py           # v9 schema (source of truth)
data_module/sentinel_data/export/chunker.py                        # chunk_export()
data_module/sentinel_data/export/export.py                         # SentinelDatasetExport
data_module/sentinel_data/cli.py                                   # sentinel-data CLI (incl. _run_split with dedup_group fix)
data_module/sentinel_data/splitting/leakage_auditor.py              # find_leaks() (text-shingle Jaccard)
data_module/sentinel_data/splitting/dedup_enforcer.py              # majority-wins enforcer
data_module/sentinel_data/preprocessing/deduplicator.py            # Level-3 STUB at line 73
data_module/sentinel_data/verification/gate.py                      # T0→VERIFIED, T2+no-reps→PROVISIONAL
data_module/sentinel_data/labeling/merger.py                        # tier-precedence merge

# Source (ml consumer)
ml/src/datasets/sentinel_dataset.py                                # 5-tuple loader with 3 hard gates
ml/src/datasets/collate.py                                         # sentinel_collate_fn
ml/src/training/trainer.py                                         # 8-site swap to SentinelDataset
ml/src/preprocessing/graph_schema.py                               # 18-line shim (re-exports from sentinel_data)

# Data artifacts
data_module/data/exports/sentinel-v3-smartbugs-2026-06-13/         # **the v3 export** (5+5 shards, hash verified, post-DoS-patch)
data_module/data/dedup_groups_graph_hash.json                       # 12,678 unique groups (graph-hash + L3 text-hash)
data_module/data/splits/v3/                                        # **active splits**: 18,596/1,983/1,914, 0% leakage, 0 DoS+Reentrancy overlap

# Logs + checkpoints
ml/logs/GCB-P1-Run11-v2deduped-20260613.log                         # Run 11 plain log (PAUSED, ep1 F1=0.3293)
ml/checkpoints/GCB-P1-Run11-v2deduped-20260613_best.pt             # Run 11 ep1 checkpoint (resumable)
ml/logs/run12_launch_2026-06-13.log                                 # **Run 12 launch log (nohup stdout)**
ml/logs/GCB-P1-Run12-v3dospatched-20260613/                          # **Run 12 structured log** (alerts/epoch_summary/step_metrics.jsonl)
ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_best.pt             # **Run 12 current best** (will appear after ep1)
```

---

## v2 → v3 transition (2026-06-13)

**What changed:**

| | v2 (Run 11) | v3 (Run 12) |
|---|---|---|
| Export dir | `sentinel-v2-baseline-2026-06-12/` | **`sentinel-v3-smartbugs-2026-06-13/`** |
| Splits | 18,458 / 1,996 / 1,902 | **18,596 / 1,983 / 1,914** |
| Sources | 2 (DIVE + SolidiFI) | **3 (DIVE + SolidiFI + SmartBugs Curated)** |
| Total contracts | 22,356 | **22,493** (+137 SmartBugs) |
| DoS=1 | 3,756 (with 2,655 DoS+Reentrancy overlap) | **1,101 (0 DoS+Reentrancy overlap)** |
| Dedup groups | 12,577 (graph-hash only) | **12,678 (graph-hash + L3 text-hash; 83 L3 groups applied)** |
| DoS loss weight | (not set) | **1.0** (user override from 0.5 default) |
| SentinelDataset speedup | 1.94s init | **0.84s init** (num_nodes from shard_index + hash cache) |
| Run state | Run 11 ep1 F1=0.3293, WSL crashed ep2 | **Run 12 launched fresh, PID 230342, ep1 in progress** |

**Why it changed (3 events on 2026-06-13):**
1. **SmartBugs Curated ingestion** added 137 contracts (parser, crosswalk fix for `front_running→TransactionOrderDependence`, T1 confidence tier)
2. **L3 text-hash dedup** applied to 83 of 147 L3 groups (consistent labels only; 64 conflicting groups skipped to avoid false-positive merges)
3. **DoS/Reentrancy co-occurrence patch** — manual data audit revealed the patch was documented but never executed. v3 export had un-patched 3,756 DoS labels (2,910 train). Applied patch to DIVE source labels (2,655 contracts zeroed where DoS+Reentrancy both=1), re-merged, re-split, re-exported.

**Result: 0 DoS+Reentrancy overlap, DoS dropped to 1,101 (845 train), other class counts unchanged, 598/27/0 tests pass.**

**Full audit trail:** `data_module/temp/pre-run12-fixes-2026-06-13.md` (the 4-step plan A-D), `data_module/temp/data-source-addition-plan-2026-06-13.md` (future data source additions), `~/.claude/projects/.../memory/project_dos_patch_2026-06-13.md` (DoS patch discovery + fix), `~/.claude/projects/.../memory/project_run12_launch.md` (Run 12 launch context + monitoring).

---

## Appendix A — Reading order for new contributors (v3-aware)

1. **This file** (5 min) — high-level architecture (v2→v3 state above)
2. **`data_module/DATA_MODULE_AUDIT.md`** §"Pipeline Architecture" + §"Source Datasets" (10 min) — what runs where
3. **`data_module/docs/v2-readiness-2026-06-12.md`** (10 min) — the 7 gates (historical v2 state; current v3 state in `pre-run12-fixes-2026-06-13.md`)
4. **`data_module/audit/v2_full_audit/07_data_quality_investigation.md`** §"Methodology" + §"Gap 1" (10 min) — what was checked and what wasn't
5. **`data_module/audit/v2_full_audit/06_FINAL_master_report.md`** (15 min) — the prior audit's Run 11 verdict
6. **`~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run12_launch.md`** (10 min) — **current Run 12** config + monitoring + DoS-patched data context
7. **`~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_dos_patch_2026-06-13.md`** (10 min) — the DoS patch audit (WHY v3 has 1,101 DoS instead of 3,756)
8. **`~/.claude/projects/-home-motafeq-projects-sentinel/memory/project_run11_launch.md`** (5 min) — Run 11 config + the 45% leakage story (HISTORICAL — Run 12 is the active training)

Total: ~60-70 min to be operational. Then read source code in order:
`cli.py` → `representation/graph_schema.py` → `export/chunker.py` → `verification/gate.py` → `splitting/leakage_auditor.py`.

End of architecture.
