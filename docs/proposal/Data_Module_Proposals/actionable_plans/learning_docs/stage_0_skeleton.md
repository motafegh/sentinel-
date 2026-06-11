# Stage 0 — Skeleton + Data/ Restructure

**Date:** 2026-06-09 (revised 2026-06-12 post-implementation)
**Status:** ✅ Code is built, committed, and preserved. Stage 0 was the structural decision; it is the foundation that Stages 1–4 built on top of. Reading required as a prerequisite.
**Reading time:** 20-30 minutes.
**Goal:** After this doc, you can answer all 15 items in `LEARNING_CHECKLIST.md` §"Stage 0" from memory, and explain why the structural decision (Branch B — full package split) was binding.

**What was actually built (2026-06-12 post-Stages-1–4):**
- `Data/sentinel_data/` is the full Python package; `Data/pyproject.toml` declares `sentinel-data` as a Poetry project
- 9 submodules: `ingestion/`, `preprocessing/`, `representation/`, `labeling/`, `verification/`, `splitting/`, `registry/`, `analysis/`, `export/`
- One-way dependency: `sentinel-ml → sentinel-data` (never the reverse), enforced by `Data/tests/conftest.py` and the package layout
- `Data/config.yaml`: 22 sources (5 critical-path + 12 additive v2.1 + 2 v1 extras + 3 dropped/deferred), BCCC in `deferred_sources:`
- v9 schema stub in `Data/sentinel_data/representation/graph_schema.py`: `FEATURE_SCHEMA_VERSION="v9"`, `NODE_FEATURE_DIM=12`, `NUM_EDGE_TYPES=12`, `NUM_NODE_TYPES=14`, `_MAX_TYPE_ID=13.0`
- 5 v2-readiness ADR documents (in `docs/decisions/`); ADR-0005 documents the Stage 4 verification design

---

## 1️⃣ The Problem

### What's actually wrong (high level)

You ran 9 training runs on the SENTINEL model. The F1 macro score on the latest best (Run 9, ep52) was 0.3081 tuned, 0.2965 fixed. The model is a 4-eye classifier over a GNN + GraphCodeBERT + LoRA architecture with 8 layers, 3 phases, and 4 eyes (GNN, TF, Fused, CFG). This architecture was the result of 5 audit phases and 8 audit fixes (A1–A38). It is, by all reasonable engineering standards, a competent model.

And yet it cannot break 0.31 F1.

### The BCCC deep-dive's finding (the bug you already know)

You committed `Data/Deep_Dive/BCCC-SCsVul-2024_Deep_Dive/Phase5_LabelVerification_2026-06-08/06_handover_p1_to_p4.md` documenting the Phase 1-5 work. The headline numbers:

- **89.4% of BCCC's "Reentrancy" labels are false positives.** A contract labeled "Reentrancy" in BCCC has only a 10.6% chance of actually being a reentrancy contract.
- **86.9% of BCCC's "CallToUnknown" labels are false positives.** Similar pattern.
- **99% of contracts labeled "DoS" are also labeled "Reentrancy"** — because BCCC stores one .sol under multiple class folders and OR-s the labels.
- **38.8% of BCCC contracts are duplicates** of each other (same content, different folder assignments).
- **The model learned a "complexity proxy"** — its top-3 features for all 10 classes are `complexity → visibility → uses_block_globals`, in the same order, regardless of the class. The class-specific diagnostic features (`return_ignored` for UnusedReturn, `external_call_count` for Reentrancy, `uses_block_globals` for Timestamp) show **zero per-class elevation** in gradient saliency (per `project_interpretability.md` Phase 2 L4 finding).

### Why this happens (root cause)

BCCC-SCsVul-2024 was assembled by putting each `.sol` file into a folder named after the class it (sometimes) has. The folder assignment is the label. So `BCCC-SCsVul-2024/Source Codes/Reentrancy/foo.sol` means "this file is labeled Reentrancy." But many of these files don't actually have a reentrancy vulnerability — they were put in the folder for other reasons (the author was grouping similar files, or the folder was a catch-all, etc.).

The model is then trained on these labels. With 89% FP, the model gets a noisy signal. To minimize loss, the model finds the only **consistent** signal in the data: contract complexity (longer, more complex contracts are more likely to be in any class folder). It learns "if complex, predict positive" — which gets ~50% accuracy on the test set, because the test set has the same noise. **The model isn't learning to detect reentrancy; it's learning to detect BCCC's folder structure.**

### The 3 branches you considered

You, with the friend's input, considered 3 options:

| Branch | Description | Why it won/lost |
|---|---|---|
| **A. Keep data in `ml/`, just clean the labels** | Replace BCCC labels with the 24,021 verified v1.4 labels. Don't restructure the package. | **Lost.** This treats the symptom (bad labels) not the cause (data+model coupled). Even with clean labels, future data sources (ScaBench, Web3Bugs, etc.) would re-introduce the same coupling. The 14-day BCCC debugging session would repeat on the next bad source. |
| **B. Full package split** ⭐ | Create a new `sentinel-data` package as a sibling of `sentinel-ml`. Sentinel-data owns the entire data pipeline (raw → verified → exported). Sentinel-ml consumes the exports. **No shared code between them.** Sentinel-ml never sees a raw contract. | **Won.** Hard boundary = no future bug can be silent. The model side can't accidentally change feature extraction. The data side can't accidentally reference model code. Any future dataset bug is caught in the data layer, not after the model has been silently training on bad features. |
| **C. Data preprocessing as a sidecar script** | Keep all data in `ml/`, but isolate the preprocessing into a separate "sidecar" script that's only invoked by `train.py` (not imported). | **Lost.** This is a halfway measure. The "sidecar" still lives in `ml/`, still has access to the model's internals (if anyone calls a function from it), still has the same coupling. The boundary is nominal, not enforced. |

The binding decision (in the integration proposal §1 + the v1 first_proposal_data_module.md that started this work) is **Branch B**. The build is the work to make Branch B real.

### Why this stage is called Stage 0 (not Stage 1)

The build is structured as 9 stages, but **Stage 0 is the most important** because it's the structural decision. Once the package boundary is committed, every later stage is **constrained by it**. Stage 1 (ingestion) can only put data in `sentinel_data.ingestion.*`; it cannot reach into `ml/src/models/`. Stage 7 (seam swap) can only delete code from `ml/` after the regression test passes; it cannot leave the boundary unverified.

If Stage 0 is wrong, every later stage is wrong. If Stage 0 is right, every later stage has a clear contract. This is why the plan allocates 4-5 working days (Jun 9-13) just to making the boundary real.

---

## 2️⃣ The Solution

### The package boundary (the design decision)

`Data/pyproject.toml` declares:

```toml
[tool.poetry]
name = "sentinel-data"
packages = [{include = "sentinel_data"}]
```

That's it. One line of package declaration. The `sentinel-ml` package's `pyproject.toml` (at `ml/pyproject.toml`) does NOT declare a dependency on `sentinel-data` (yet — that's added in Stage 7). The direction is **one-way**: `sentinel-ml → sentinel-data`, never `sentinel-data → sentinel-ml`.

This is enforced at install time by `poetry show --tree` in `Data/`. The Stage 0 test `test_no_sentinel_ml_dependency` asserts this. If anyone ever adds `sentinel-ml = "..."` to `Data/pyproject.toml`, that test fails loud.

### The 9 submodules (why this many)

The package has 9 submodules, one per pipeline stage. Each takes the previous stage's output as input. The DAG is enforced by DVC (`Data/dvc.yaml`):

```
ingest → preprocess → represent → label → verify → split → register → analyze → export
                                                                                ↓
                                                                       sentinel-ml (consumer)
```

Each submodule has its own `__init__.py` (empty in Stage 0; the real code comes in Stages 1-7). Each has a corresponding DVC stage with placeholder commands.

**Why 9 and not 3 or 15?** Three branches were considered:

| Sub-modules | Pros | Cons |
|---|---|---|
| 3 (data, label, export) | Simple, fewer files | "Data" is too vague — ingest, preprocess, represent are different concerns. Verification and registry would get lost. |
| **9 (current)** | One per pipeline stage. Each has a single responsibility. | More files. But each is small (Stage 0 stubs are ~30 lines each). |
| 15 (one per source connector + per crosswalk + per verification component) | Maximum separation | Over-fragmented. Connectors are similar; crosswalks are just data files; the verification components share infrastructure. |

9 is the sweet spot: each submodule is a real concern, and Stages 1-7 each own exactly one (plus Stage 0 setting up the skeleton).

### The schema stub (the deferred-decision pattern)

Stage 0 ships a **stub** of `representation/graph_schema.py` that holds the active v9 schema constants by value. Why a stub and not the real port?

**The problem with porting real code in Stage 0:** The real `graph_extractor.py` in `ml/src/preprocessing/` is currently in use by Run 9 training. Moving it to `sentinel_data/representation/` would require either (a) breaking the import path (Run 9 breaks), (b) keeping both copies in sync (defeats the purpose), or (c) doing the port + regression test in Stage 0 (Stage 0 grows to 2 weeks).

**The stub pattern:** Stage 0 ships constants + a `NotImplementedError` for the extractor. The constants are correct (verified against `ml/src/preprocessing/graph_schema.py:161,175,218` on 2026-06-08). The extractor is empty. The dual-import test (old `ml/src/preprocessing/graph_extractor` vs new `sentinel_data.representation.graph_extractor`) is **deferred to Stage 2**, where the regression test is the gate. By then, Run 9 is done or paused, so breaking the import is safe.

**Why "v9 not v8"** — the v9 detail is critical. The proposal §2 said v8 (11 features, 11 edge types, 13 node types). But the live `ml/src/preprocessing/graph_schema.py:161,175,218` says v9 (12 features, 12 edge types, 14 node types). The v9 schema adds `feat[11] = in_unchecked_block`, `CFG_NODE_ARITH=13`, `EXTERNAL_CALL=11` self-loop. The stub **must** say v9 or the Stage 2 byte-identical regression test fails immediately. This was the #1 risk per `AUDIT_PATCHES.md` F1.

### The config (22 sources, correctly tiered)

`Data/config.yaml` is the single source of truth for "which sources, which tier, which connector, which crosswalk." It has:

- **8 Tier-1 (gold) sources** — human-audited, peer-reviewed, or mathematically certain labels
  - ScaBench, SmartBugs Curated, SolidiFI, DIVE, FORGE, Web3Bugs, DeFiHackLabs, solidity-defi-vulns
  - Wait, that's 8. Plus Bastet = 9. (Bastet was originally Tier 2 in the first draft; friend promoted to Tier 1.)
- **2 Tier-2 (clean) sources** — OpenZeppelin Contracts, OpenZeppelin Ethernaut (the strongest practical negative pool)
- **3 Tier-2 (silver/gold) sources** — ScrawlD (5-tool majority voting), Code4rena (human auditors), DeFi Hacks REKT (verified real exploits)
- **3 Tier-3 (structural) sources** — SC-Bench, DeFiVulnLabs, plus SmartBugs Curated was moved up to T1
- **2 Tier-3 (bronze) sources** — SmartBugs Wild (97% FP warning per friend's research), Slither-Audited (tool-only, conservative)
- **2 Tier-4 (bronze) sources** — DISL, ReentrancyStudy (URLs TBD, deferred)
- **2 v1 extras** — Messi-Q, Zenodo 16910242 (CLEAR)
- **BCCC in `deferred_sources:`** — not consumed by v2, preserved at legacy path for reference

Why this distribution? **Quality scales with tier; volume scales inversely.** Tier 1 sources have 100s of verified labels each. Tier 4 sources have 100,000s of unverified labels. The training corpus is mostly Tier 1+2 (high quality) for the gold standard, augmented with Tier 3 (benchmarks for evaluation), plus Tier 4 (volume for pretraining). The friend called this the "Strategy D" in `datasources_suggestions.md`.

**3 sources are enabled by default** (ScaBench, SmartBugs Curated, DeFiHackLabs). The plan's first 5 stages use these 3 for end-to-end testing. Stage 1 will add the other 19 as their connectors are built.

### The 8 already-fixed bugs (the preservation list)

Per `AUDIT_PATCHES.md` F2–F6, F11, F29, 8 bugs in `ml/src/` are already fixed. **Stage 0 does NOT re-fix them.** The fix is preserved through Stage 2 (port) by the 36-issue pre-Run-8 audit regression test, and through Stage 7 (seam swap) by the dual-path byte-identical test.

| Bug | Fix location | Why it's preserved |
|---|---|---|
| **A20** label=0 hardcode | `ast_extractor.py:290,342,395` | Stage 2 regression test asserts the fix |
| **A9** `now` keyword miss | `graph_extractor.py:587-605` | Stage 2 regression test asserts the fix |
| **A15** def_map by name | `graph_extractor.py:1147-1179` | Stage 2 regression test asserts the fix |
| **A34** prefix sort dim | `sentinel_model.py:356,367` | Stage 2 regression test asserts the fix |
| **A38** NaN before backward | `trainer.py` | Stage 7 (Run 11 launch) uses it |
| Resume overwrite | `trainer.py:383,1184,1206,1212` | Stage 8 (Run 11 launch) uses full-resume default |
| `_compute_return_ignored` | `graph_extractor.py` | Stage 4 verification uses `feat[7]` |
| **EMITS edge** | `graph_extractor.py` (Interp-6) | **OPEN; fixed in Stage 7** |

The Stage 0 stub **does not re-implement** any of these. `graph_extractor.py` in `sentinel_data/representation/` raises `NotImplementedError`. The 7 fixes are in `ml/src/` and stay there until Stage 7 (seam swap) deletes them. The `ADR-0002` documents this state for downstream reference.

### The 3 still-open bugs (to be closed in Stage 7)

| Bug | Location | Stage that closes it |
|---|---|---|
| Predictor tier threshold | `predictor.py:150,168,752` | **Stage 7** — the tier bug is hidden by the seam swap |
| EMITS edge | `graph_extractor.py` (Interp-6) | **Stage 7** — fix during port |
| CALL_ENTRY cross-function for external | `graph_extractor.py:1001` (self-loop only) | **Stage 7 (or v2.1)** — full cross-function edge is post-Run-11 |

The Stage 0 plan intentionally **does not** close these. They're a Stage 7 concern. Stage 0 ships a skeleton that preserves the current state.

---

## 3️⃣ The Broader Context

### What Stage 0 enables downstream

| Stage | What it builds on Stage 0 | What breaks if Stage 0 is wrong |
|---|---|---|
| Stage 1: Ingestion | Adds 5 source connectors (Git, HF, Zenodo, Etherscan, Manual). Reads from `config.yaml` source list. | If `config.yaml` is wrong (e.g. wrong URL), Stage 1 downloads the wrong dataset. The 27 tests don't catch this until Stage 1 runs. |
| Stage 2: Representation | Ports `graph_extractor.py` to `sentinel_data.representation`. Uses the v9 schema stub as the source of truth. | If the v9 schema stub has wrong constants, the byte-identical regression test fails immediately. The fix is to update the stub to match `ml/src/preprocessing/graph_schema.py`. |
| Stage 3: Labeling | Authoring 17 crosswalk YAMLs + 17 parsers. Reads from `config.yaml` source descriptions. | If the source list in `config.yaml` is wrong (e.g. tier assignments), the crosswalks are tiered wrong. |
| Stage 4: Verification | Runs the 6 verification components on each labeled dataset. Uses the 36-issue pre-Run-8 audit regression test. | If the 8 already-fixed bugs are not preserved through the port, Run 11 fails silently. |
| Stage 5: Splitting + Registry | Writes to SQLite catalog. Uses `dvc.yaml` for orchestration. | If the 9-stage DVC DAG is wrong, downstream stages don't re-run correctly. |
| Stage 6: Analysis | Computes `complexity_proxy_risk.md`. Uses `representation/.rep.json` sidecars from Stage 2. | If the v9 schema stub has wrong NODE_FEATURE_DIM, the per-class feature distribution is computed against the wrong feature count. |
| Stage 7: Seam Swap | Deletes `ml/src/preprocessing/graph_extractor.py` and `ml/src/datasets/dual_path_dataset.py`. Closes 2 of 3 open bugs. | If the dual-path test fails, the seam swap breaks Run 9 (or Run 11, whichever is live). |
| Stage 8: Run 11 launch | Uses the v9-trained model + v2 corpus. | If the v9 schema is wrong, the model's classifier head receives wrong input shape. |

### What changes if you change Stage 0

- **Change the schema stub constants** (e.g. revert to v8) → Stage 2 byte-identical test fails → Stage 7 seam swap blocked → Run 11 cannot launch
- **Remove a source from `config.yaml`** → its crosswalk YAML is unreachable → that class has fewer positives in the v2 corpus → per-class F1 distribution changes
- **Move BCCC from `deferred_sources:` to regular `sources:`** → the 89% FP noise re-enters the training pipeline → all of v2's effort is wasted
- **Add `sentinel-ml` to `Data/pyproject.toml` deps** → `test_no_sentinel_ml_dependency` fails → the build is blocked → the entire premise (one-way dependency) is violated
- **Skip the 8 already-fixed bugs preservation** → regression test fails in Stage 2 → Run 11 trains on the buggy old code → silent regression

### What stays the same no matter what

- The package boundary (one-way dependency) — this is the binding decision, not negotiable
- The 8 already-fixed bugs preservation list — these are part of the codebase contract
- The BCCC deferral — the 89% FP problem is structural; we work around it, we don't fix it
- The v9 schema — this is the live training schema; we don't change it during v2

### The "What NOT to fix" table (the most important doc in Stage 0)

This table is the first thing every later-stage contributor reads. It says: "these 8 bugs are already fixed; do not re-fix them; the regression test guards them." Without this table, the next contributor (or future-you) might "helpfully" re-fix A9 (the `now` keyword) because they think it's still broken — and introduce a different bug.

---

## 4️⃣ Verification — How to know Stage 0 is right

The 15 final exit criteria (all PASS as of 2026-06-09):

| # | Check | What it proves |
|---|---|---|
| 1 | `docs/legacy/bccc_deep_dive/` exists; `Deep_Dive/` doesn't | BCCC is preserved at the new location, not lost |
| 2 | `poetry install` works | The package boundary is buildable |
| 3 | `FEATURE_SCHEMA_VERSION == 'v9'` | The schema stub matches the live training schema |
| 4 | `(12, 12, 14) == (NODE_FEATURE_DIM, NUM_EDGE_TYPES, NUM_NODE_TYPES)` | All 3 dimensions correct |
| 5 | `sentinel-data --help` exits 0 with 9 stages | CLI is wired correctly |
| 6 | `sentinel-data run --dry-run` lists all 9 stages | DAG is built correctly |
| 7 | `dvc status` clean | DVC orchestration is initialized |
| 8 | 27/27 tests pass | All structural assertions hold |
| 9 | 9 subpackage dirs exist | Package layout matches proposal §2 |
| 10 | README, architecture, ADR-0001, ADR-0002 exist | Documentation is committed |
| 11 | dvc.yaml, config.yaml (sqlite), Dockerfile (bookworm) exist | Infrastructure is configured |
| 12 | `poetry show --tree | grep -i sentinel-ml` returns empty | The one-way dependency is enforced |
| 13 | `contracts_clean_v1.4.csv` reachable at new path | BCCC legacy is preserved |
| 14 | `_schema_constants.md` and `_schema_version_registry.json` exist | Schema is documented + versioned |
| 15 | `config.yaml` lists 22 sources + BCCC deferred | The source list is complete + correct |

If any of these fail, Stage 0 is NOT done and Stage 1 cannot start.

---

## 5️⃣ The "got it" checklist

Before we move to Stage 1, you should be able to answer (without looking at this doc):

1. **Why v2 exists** (BCCC's 89% FP / 86.9% FP / 99% co-occurrence / 38.8% dup / complexity proxy) — and why the v2 fix is "separate the data package," not "clean the labels."

2. **Why the 3 branches were A (keep data in ml), B (full package split), C (sidecar script)** — and why B won (hard boundary = no future bug can be silent).

3. **Why the package is rooted at the existing `Data/`** (continuity with the deep-dive work, BCCC legacy at `docs/legacy/bccc_deep_dive/`, root README cross-link).

4. **Why the schema stub ships with v9 not v8** (the live training schema is v9; the byte-identical test in Stage 2 would fail otherwise).

5. **Why BCCC is in `deferred_sources:` not regular `sources:`** (the 89% FP problem is structural; we work around it).

6. **Why the 8 already-fixed bugs are preserved, not re-fixed** (silent regression risk; the 36-issue regression test guards them in Stage 2).

7. **Why the 3 still-open bugs are NOT closed in Stage 0** (scope discipline; Stage 7 closes them).

8. **What the 15 final exit criteria prove** (the skeleton is correct, the boundary is enforced, the schema matches, the tests pass).

9. **What changes if you modify Stage 0 after Stages 1-7 are built** (downstream stages break; v2 corpus changes; Run 11 fails silently).

10. **What stays the same no matter what** (the one-way boundary, the preservation list, the BCCC deferral, the v9 schema).

If you can answer all 10, Stage 0 is mastered and we can move to Stage 1.

---

## 6️⃣ What to read next

- **LEARNING_CHECKLIST.md** §"Stage 0" — 15 specific questions to test your understanding
- **Sentinel_v2_Data_Module_Integration_Proposal.md** §2 (module structure) + §5 (build order) — the design contract
- **AUDIT_PATCHES_applied_2026-06-08.md** §0 (F1-F30) — the 30 verified facts this plan was built on
- **project_run8_audit_findings.md** (in MEMORY) — the original audit that found the complexity proxy problem
- **project_data_pipeline_audit.md** (in MEMORY) — the 99% co-occurrence + 38.8% dup numbers

When you're ready, say "Stage 0 is mastered — let's move to Stage 1."
