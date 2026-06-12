# Phase C2 — Stage 7 (Export + Seam Swap) Deep Audit

**Sessions:** 1 (~2.5h)
**Output:** `v2_full_audit/04_phase_c2_stage_7_export_audit.md`
**Status:** PENDING (gated on Phase C1 DONE)

> **Apply the [Hostile Verification Protocol](../../00_INDEX.md#hostile-verification-protocol-applies-to-all-phases).** The seam-swap is the single most dangerous code in the audit. For each check: actually run the comparison, don't just describe it. The dormant `_backup_pre_seam_swap_2026-06-12_graph_schema.py` is a critical-state artifact — diff it against live, don't just acknowledge it.

---

## Goal

Audit Stage 7 — the most complex stage in the build. Stage 7 is the export + seam swap that physically removes the old `ml/src/preprocessing/*` and replaces `dual_path_dataset.py` with the new `sentinel_dataset.py`. A bug here breaks Run 9 and silently blocks Run 11.

---

## What this phase touches

### Stage 7 — export (7 files, ~900 LOC)

| File | LOC | What |
|---|---|---|
| `export/__init__.py` | 27 | Module docstring |
| `export/chunker.py` | 210 | Per-shard orchestration; manifest write |
| `export/export.py` | 141 | `SentinelDatasetExport` class — consumer-facing API |
| `export/graph_writer.py` | 102 | PyG Batch → sharded `.pt` |
| `export/token_writer.py` | 95 | Tensor → sharded `.pt` |
| `export/label_writer.py` | 150 | parquet writer for labels |
| `export/metadata_writer.py` | 200 | parquet writer for metadata |
| `export/format_schema/` | dir | The v1 spec (per plan §3.8) |

### Seam swap artifacts

| File | Why |
|---|---|
| `representation/_backup_pre_seam_swap_2026-06-12_graph_schema.py` | Snapshot taken 2026-06-12 — was the seam swap done partially, then reverted? |
| `representation/graph_schema.py` (live) | If different from the backup, the swap is partial. |

### CLI + tests for Stage 7

| File | Why |
|---|---|
| `cli.py:_STAGE_FN` dispatch (line ~679) | Does `export` map to a real handler or to the stub? |
| `cli.py:223-229` (the prior Stage 3 stub) | Per the README, the labeling CLI is a stub — verify |
| `cli.py:Stage 7` handler (if it exists) | Does it call `SentinelDatasetExport` correctly? |
| `tests/test_export/` (5 test files per the `tests/` listing) | Do they actually run? Phase A counted pass/fail; this phase reviews the assertions. |

### Cross-package bridges (read-only, but their state matters)

- `ml/src/datasets/dual_path_dataset.py` — does it still exist? If yes, the seam swap is not done.
- `ml/src/preprocessing/{graph_extractor,graph_schema}.py` — do they still exist? If yes, the seam swap is not done.
- `ml/src/inference/predictor.py:150,168,752` — has the tier threshold bug been fixed?

---

## Tasks (ordered, each with exit condition)

### C2.1 — Verify the seam-swap state (3 separate checks)

**C2.1.a — Compare backup vs live graph_schema.py:**

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module/sentinel_data && diff representation/_backup_pre_seam_swap_2026-06-12_graph_schema.py representation/graph_schema.py'
```

**Exit condition:** the diff is captured. Three possible outcomes:
- Identical (0 lines) → seam swap not done; backup is a paranoid copy
- Live has FEWER lines than backup → seam swap was reverted
- Live has MORE lines than backup → backup is stale, ignore it

**C2.1.b — Check the `ml/` paths still exist:**

```bash
wsl -- bash -c 'cd ~/projects/sentinel && ls -la ml/src/datasets/dual_path_dataset.py 2>/dev/null; ls -la ml/src/preprocessing/graph_extractor.py 2>/dev/null; ls -la ml/src/preprocessing/graph_schema.py 2>/dev/null'
```

**Exit condition:** each file is either PRESENT (seam swap incomplete) or DELETED (seam swap done).

**C2.1.c — Check predictor tier threshold fix:**

```bash
wsl -- bash -c 'cd ~/projects/sentinel && sed -n "145,175p" ml/src/inference/predictor.py && echo "---" && sed -n "748,758p" ml/src/inference/predictor.py'
```

Per `00_EXECUTIVE_SUMMARY.md:188` and Stage 7 plan §7.8, the fix is:
- `_format_result` consults `self.thresholds` per class
- "confirmed" tier is per-class-tuned threshold (not hardcoded 0.55)

**Exit condition:** fix verdict — FIXED / OPEN / PARTIAL.

**Section verdict:** write a 3-row table in the output doc with the state of each.

### C2.2 — Read the Stage 7 plan and exit criteria

```bash
wsl -- bash -c 'cat ~/projects/sentinel/docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md'
```

Build a check-list of:
- 11 tasks (7.1 through 7.11) and their exit conditions
- 16 final exit criteria
- 6 risk items
- All design decisions (D-7.1 through D-7.6)

**Exit condition:** design-decision + exit-criteria inventory table in working notes.

### C2.3 — Per-file review of export/

Same format as Phase C1.

For each of the 7 export/ files, build a review table:
- Public API matches what `cli.py` (or future `sentinel_dataset.py`) needs
- Error handling
- Path handling
- Determinism
- Format spec compliance (`format_schema/v1.yaml` if it exists)

**Special focus areas:**

- **`export.py` (`SentinelDatasetExport` class)**:
  - Does `verify_artifact_hash()` actually re-hash or trust the manifest?
  - Does it expose `graphs_path`, `tokens_path`, `labels_path`, `metadata_path`, `shard_index`, `manifest`?
  - Does it handle the case where one shard is missing?

- **`chunker.py`**:
  - Does the manifest correctly map `contract_id → shard_number`?
  - Are all 4 writers (`graph_writer`, `token_writer`, `label_writer`, `metadata_writer`) called?
  - Does it support a `--limit` smoke-test mode?

- **`format_schema/v1.yaml`**:
  - Does the spec file exist? (Per Phase A check.)
  - Does it cover all 4 file types and the required manifest fields?
  - Does it have a `version` field (so future v1.1 bumps are a new file, not an edit)?

**Exit condition:** 7 review tables (one per file, with format_schema/v1.yaml as a 1-row check).

### C2.4 — Test review (export tests)

The `tests/test_export/` folder has 5 test files. For each, check:

- Does it cover the happy path?
- Does it cover the missing-shard case?
- Does it cover the schema-version-mismatch case?
- Does it run in <60s? (Per Stage 7 plan, the export is supposed to be testable in <60s for a 1.0 GB sharded artifact.)

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && for f in tests/test_export/test_*.py; do echo "===$f==="; grep -c "def test_" $f; done'
```

**Exit condition:** test coverage table per file.

### C2.5 — Critical seam-swap bugs (per `08_stage_7_export_seam.md` "What NOT to fix" table)

Per the plan, these are bugs the seam swap must NOT regress:

- **A9** `now` keyword — regression test guards it
- **A15** def_map by name — regression test guards it
- **A20** label=0 hardcode — regression test guards it
- **A34** prefix sort dim — regression test guards it
- **A38** NaN before backward — regression test guards it
- **Resume overwrite** — full-resume default
- **EMITS edge bug** — Stage 7 MUST FIX (per plan §7.8). MEMORY says it's already fixed in the data module; the audit should verify.
- **Predictor tier threshold** — Stage 7 MUST FIX. Verified in C2.1.c.
- **CALL_ENTRY cross-function for external** — partial fix preserved (self-loop).

For each, find the regression test:

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && grep -rln "now\|A9\|A15\|A20\|A34\|A38\|EMITS\|def_map\|return_ignored\|prefix.*sort\|label.*hardcode\|nan.*backward\|CALL_ENTRY" tests/ | head -20'
```

**Exit condition:** 11-row table (one per bug) with the regression test path and verdict (PRESENT/ABSENT/STALE).

### C2.6 — The 5 critical-path source parsers (re-verify from Phase A)

Phase A found the state of DeFiHackLabs, Web3Bugs, SmartBugs Curated parsers. This phase does the **deep review** of any that were FOUND (or marks MISSING as a Run 11 blocker).

For each parser that exists:
- Does it produce the correct crosswalk output?
- Does it write to `data/labels/<source>/` correctly?
- Does its test cover the multi-label case (DIVE) and single-label case (SmartBugs)?

For each parser that's MISSING:
- Is it needed for Run 11? (Per MEMORY critical-path list.)
- If yes, that's a Run 11 blocker.

**Exit condition:** 3-row table with deep-review verdicts.

### C2.7 — Output doc structure

Author `v2_full_audit/04_phase_c2_stage_7_export_audit.md` with:

1. **Executive summary** — seam-swap state + blocker list
2. **Seam-swap state (3 checks)** — from C2.1
3. **Stage 7 plan compliance** — D-X.Y + 16 exit criteria from C2.2
4. **export/ per-file review** — 7 review tables from C2.3
5. **format_schema/v1.yaml check** — 1-row table from C2.3
6. **Test coverage review** — 5-file table from C2.4
7. **Critical bug preservation** — 11-row table from C2.5
8. **Critical-path source parsers** — 3-row table from C2.6
9. **Findings inventory** — `FINDING-C2:N` numbered list with severity
10. **Run 11 blockers from C2** — items that must be fixed before Run 11

---

## What this phase will NOT touch

- The actual seam-swap DELETE operation (this is a planning audit, not a refactor)
- `ml/src/inference/predictor.py` — that's a code fix, not an audit item; Phase D verifies it
- Round-trip integration test (Phase D)

---

## Required inputs

- `v2_full_audit/01_phase_a_foundation_recon.md` — Phase A findings (parser state, test pass rate)
- `v2_full_audit/02_phase_b_prior_audit_compliance.md` — items still open
- `v2_full_audit/03_phase_c1_stages_5_6_audit.md` — cross-stage contracts
- `docs/proposal/Data_Module_Proposals/actionable_plans/08_stage_7_export_seam.md` — Stage 7 plan

## Outputs

- `v2_full_audit/04_phase_c2_stage_7_export_audit.md` (consumed by Phase E)
- Updated todo list

---

## Exit criteria checklist

- [ ] 3 seam-swap state checks done (backup diff, ml/ paths, predictor fix)
- [ ] Stage 7 plan inventory built (11 tasks + 16 criteria + 6 risks + 6 design decisions)
- [ ] All 7 export/ files reviewed
- [ ] format_schema/v1.yaml exists (or is documented as missing)
- [ ] All 5 export test files reviewed
- [ ] All 11 critical bugs have regression-test verdicts
- [ ] All 3 critical-path source parsers reviewed (or marked missing)
- [ ] Output doc authored with all 10 sections
- [ ] All findings numbered as `FINDING-C2:N`
- [ ] Run 11 blockers from C2 identified
