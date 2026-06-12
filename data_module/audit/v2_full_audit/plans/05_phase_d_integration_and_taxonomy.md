# Phase D — Integration, Cross-Cutting, and the Two-Taxonomy Question

**Sessions:** 1 (~2.5h)
**Output:** `v2_full_audit/05_phase_d_integration_and_taxonomy.md`
**Status:** PENDING (gated on Phase C2 DONE)

> **Apply the [Hostile Verification Protocol](../../00_INDEX.md#hostile-verification-protocol-applies-to-all-phases).** Phase D's core question (the two-taxonomy divergence) is decided by a **3-way diff** with **evidence on all three sides** — don't accept a single-source quote as proof. Every gate in the Run 11 readiness matrix needs a runnable command that produces the verdict, not a hand-wave.

---

## Goal

Answer the questions the per-file audits can't: does the data module actually work end-to-end, and which design ambiguities will silently break Run 11?

The single most important task in this phase is the **two-taxonomy decision** (D.4 below). The README at `data_module/README.md:218-223` flags a divergence between the representation class order and the labeling class order, and there's no ADR and no test that pins the correct behavior. This must be resolved (or explicitly deferred) before Run 11.

---

## What this phase touches

- The 2 taxonomies in source:
  - `representation/graph_schema.py:73-84` (re-export of `ml/src/preprocessing/graph_schema.py:73-84`)
  - `labeling/schema/taxonomy.yaml:21-159`
- The v9 model checkpoint (read-only): `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt`
- The trainer: `ml/src/training/trainer.py` (just the class-order reference)
- `ml/src/datasets/dual_path_dataset.py` (does it still exist?)
- `ml/src/inference/predictor.py` (the predictor tier fix verification)
- The ScaBench fixture (smoke test): `data/raw/solidifi/` or wherever the integration test data lives
- `dvc.yaml` (does `dvc repro` work end-to-end?)
- All 5 integration tests: `tests/test_integration_*.py`
- `docker/Dockerfile.data` (does it build?)

---

## Tasks (ordered, each with exit condition)

### D.1 — The two-taxonomy divergence: a one-shot proof of correctness

**The question:** when a v2 export is written and a v9-trained model loads it, which class order is the contract?

**Inputs to read:**

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module/sentinel_data && echo "===representation order===" && sed -n "73,90p" representation/graph_schema.py && echo "===labeling order===" && sed -n "20,60p" labeling/schema/taxonomy.yaml'
wsl -- bash -c 'cd ~/projects/sentinel && echo "===ml trainer CLASS_NAMES===" && grep -n "CLASS_NAMES\|_TRAINER_CLASS_NAMES" ml/src/training/trainer.py | head -10 && echo "===v9 checkpoint metadata===" && python3 -c "import torch; ckpt = torch.load(\"ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt\", map_location=\"cpu\", weights_only=False); print({k: ckpt.get(k) for k in [\"class_names\", \"CLASS_NAMES\", \"config\"] if k in ckpt})" 2>&1 | head -20'
```

**Build a 3-way diff:**

| Source | Order | Index of "Reentrancy" | Index of "NonVulnerable" |
|---|---|---|---|
| `representation/graph_schema.py:73-84` | (read) | … | … |
| `labeling/schema/taxonomy.yaml:21-159` | (read) | … | (no NonVulnerable slot) |
| v9 checkpoint class order | (read) | … | … |

**The two scenarios:**

- **Scenario A — Representation order matches checkpoint:** v2 exports should write labels in representation order. The merger in `labeling/merger.py` must reindex. The crosswalk YAMLs must be reindexed too. `taxonomy.yaml` becomes a documentation-only thing (the string→index map is read from `graph_schema.py`).
- **Scenario B — Labeling order is the contract:** v2 exports should write labels in labeling order. The graph schema in `representation/` must be re-ordered to match. The v9 checkpoint class indices must be re-mapped at inference time. `taxonomy.yaml` is the source of truth.

**Output:** write a `data_module/audit/v2_full_audit/05a_two_taxonomy_decision.md` (referenced from the Phase D output) that:
1. Lays out both scenarios with their file:line implications
2. Recommends one (with the blast radius estimated)
3. Has a placeholder for Ali's sign-off

**Exit condition:** the 3-way diff is captured; the scenario doc is drafted; `FINDING-D:1` (HIGH or CRITICAL) is opened.

### D.2 — Schema-version gate end-to-end

Per the Stage 7 plan §7.6, a v9 export loaded in a v11-trained model must raise. The gate is in `sentinel_data/registry/catalog.py:hash_artifact` (or wherever the schema-version check is).

Test the gate:

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && poetry run python -c "
from sentinel_data.registry import catalog
# Simulate a v9 export manifest
v9 = {\"feature_schema_version\": \"v9\", \"extractor_version\": \"v2.1-windowed-gcb\"}
v8 = {\"feature_schema_version\": \"v8\", \"extractor_version\": \"v2.1-windowed-gcb\"}
print(\"v9 verify (expect True):\", catalog.verify_artifact_hash(v9_export=v9))
print(\"v8 verify (expect False or warning):\", catalog.verify_artifact_hash(v9_export=v8))
" 2>&1 | tee /tmp/phase_d_schema_gate.log'
```

**Exit condition:** the gate behaves as documented. If it doesn't, that's `FINDING-D:2`.

### D.3 — DVC end-to-end on the ScaBench fixture

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && dvc repro 2>&1 | tail -50'
```

If DVC is not initialized for the data module, this will fail fast — that's a finding, not a bug to fix in this audit.

**Exit condition:** `dvc repro` runs without error, OR the failure is captured as `FINDING-D:3`.

### D.4 — CLI dispatch end-to-end

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && poetry run sentinel-data run --dry-run --from-stage represent 2>&1 | head -50'
```

The question: does `--from-stage represent` actually start at `represent` (skipping ingest+preprocess), or does it restart from the beginning?

**Exit condition:** dry-run output is captured. If `--from-stage` is broken, that's `FINDING-D:4`.

### D.5 — The 36 pre-Run-8 code-bug regression tests

Per `data_module/audit/00_EXECUTIVE_SUMMARY.md:241` and `08_stages_0_2_deep_audit.md:262-275`, there are 36 code-bug regressions (A1–A38) that the suite must guard. Some are critical:
- A9 (now keyword)
- A15 (def_map by name)
- A20 (label=0 hardcode)
- A34 (prefix sort dim)
- A38 (NaN before backward)
- A9, A15, A20, A34, A38 + resume + def_use + return_ignored (the "8 fixed bugs")

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && grep -rln "A9\|A15\|A20\|A34\|A38\|now.*keyword\|def_map\|return_ignored\|prefix.*sort" tests/ | head -30'
```

For each of the 8 fixed bugs:
- [ ] Is there a regression test?
- [ ] Does the test pass? (From Phase A pass/fail count.)
- [ ] Does the test guard the right behavior?

**Exit condition:** 8-row table with verdict per bug.

### D.6 — EMITS edge + predictor tier threshold

Two bugs that the Stage 7 plan §7.8 says MUST be fixed during the seam swap.

**EMITS edge:** MEMORY says "EMITS edges (type 3) ARE generated for 0.4.21+ contracts with `emit` keyword. BUG-H7 was already fixed." Phase C2 verified a test exists. This task confirms the test actually exercises the fix:

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && cat tests/test_representation/test_emits_fixture.py | head -50'
```

**Predictor tier threshold:** Phase C2 already checked the file. This task verifies the regression test exists:

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && find tests/ -name "*predictor*" -o -name "*tier*"'
```

**Exit condition:** 2-row table with verdicts.

### D.7 — Docker build attempt (single attempt, time-boxed)

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && docker build -t sentinel-data:audit -f docker/Dockerfile.data . 2>&1 | tail -40' &
BUILD_PID=$!
sleep 600  # 10 minutes
if kill -0 $BUILD_PID 2>/dev/null; then
    kill $BUILD_PID
    echo "BUILD TIMED OUT — recording as FINDING-D:N"
fi
wait $BUILD_PID
```

This is best-effort. Docker builds can take 15+ minutes; record the result either way.

**Exit condition:** build success/failure recorded.

### D.8 — All 5 integration tests (smoke test only)

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && for f in tests/test_integration_*.py; do echo "===$f==="; poetry run pytest $f -v 2>&1 | tail -10; done'
```

The 5 integration tests likely are: `test_integration_solidifi.py`, `test_integration_dive.py`, plus 3 more (per `tests/test_*` listing — confirm). They exercise the full pipeline against real source data.

**Exit condition:** 5-row pass/fail table.

### D.9 — Output doc structure

Author `v2_full_audit/05_phase_d_integration_and_taxonomy.md` with:

1. **Executive summary** — the two-taxonomy decision + Run 11 readiness verdict
2. **Two-taxonomy proof** — 3-way diff + scenario analysis (from D.1)
3. **Schema-version gate test** — output from D.2
4. **DVC end-to-end** — output from D.3
5. **CLI dispatch test** — output from D.4
6. **36-issue regression coverage** — 8-row table from D.5
7. **EMITS + predictor fix verification** — 2-row table from D.6
8. **Docker build** — output from D.7
9. **Integration tests** — 5-row table from D.8
10. **Findings inventory** — `FINDING-D:N` numbered list with severity
11. **Run 11 blockers from D** — items that must be fixed before Run 11
12. **Open design questions for Ali** — items the audit can document but not decide (start with the two-taxonomy question)

---

## What this phase will NOT touch

- Fixing any of the bugs (this is an audit, not a refactor)
- Launching Run 10/11
- The actual ML model training (no GPU, no torch imports beyond checkpoint read)

---

## Required inputs

- `v2_full_audit/01_phase_a_foundation_recon.md` — test counts
- `v2_full_audit/02_phase_b_prior_audit_compliance.md` — open items
- `v2_full_audit/03_phase_c1_stages_5_6_audit.md` — cross-stage contracts
- `v2_full_audit/04_phase_c2_stage_7_export_audit.md` — seam-swap state
- `ml/checkpoints/GCB-P1-Run9-v11-20260606_best.pt` (read-only, for class-order check)

## Outputs

- `v2_full_audit/05_phase_d_integration_and_taxonomy.md` (consumed by Phase E)
- `v2_full_audit/05a_two_taxonomy_decision.md` (decision doc; awaits Ali sign-off)
- Updated todo list

---

## Exit criteria checklist

- [ ] Two-taxonomy 3-way diff captured (representation vs labeling vs checkpoint)
- [ ] Two-taxonomy decision doc drafted
- [ ] Schema-version gate tested
- [ ] DVC end-to-end attempted (success or documented failure)
- [ ] CLI `--from-stage` tested
- [ ] 8 fixed-bug regression tests verified
- [ ] EMITS + predictor fix verified
- [ ] Docker build attempted (best-effort)
- [ ] All 5 integration tests run
- [ ] Output doc authored with all 12 sections
- [ ] All findings numbered as `FINDING-D:N`
- [ ] Run 11 blockers from D identified
