# Phase B — Prior Audit Compliance Check

**Sessions:** 1 (~1.5h)
**Output:** `v2_full_audit/02_phase_b_prior_audit_compliance.md`
**Status:** PENDING (gated on Phase A DONE)

> **Apply the [Hostile Verification Protocol](../../00_INDEX.md#hostile-verification-protocol-applies-to-all-phases).** For each prior-audit item: don't trust the prior verdict, re-verify with a fresh read of the code + a command output. For each "PASS" verdict in the prior audit, re-verify ~10% (sample).

---

## Goal

For every FAIL and WARN finding in the existing `data_module/audit/00_EXECUTIVE_SUMMARY.md` and `08_stages_0_2_deep_audit.md`, determine the current state: **OPEN / FIXED / DOCUMENTED-DEFER**.

This is a compliance sweep, not a re-audit. We trust the prior auditors' file:line references; we verify whether the underlying code has been changed.

---

## What this phase touches

- `data_module/audit/00_EXECUTIVE_SUMMARY.md` (P0–P3 lists, 79 items total)
- `data_module/audit/08_stages_0_2_deep_audit.md` (P0–P3 lists, ~80 items)
- The 23 prior FAIL items specifically
- The 2 CRITICAL items explicitly: REP-2, REP-7
- The 1 MEDIUM items: V-1, V-2, V-3, F-3.1, F-3.2

---

## Tasks (ordered, each with exit condition)

### B.1 — Load the prior P0 list (10 items)

From `00_EXECUTIVE_SUMMARY.md:152-164` (lines quoted for traceability):

| # | Prior ID | Issue | File:line | Current status |
|---|----|----|----|----|
| 1 | ING-1 | Pin enforcement missing | `ingestion/ingest.py:33` | OPEN / FIXED / DEFER |
| 2 | F7 / REP-7 | stale_entries() wrong sha | `representation/cache_manager.py:95` | … |
| 3 | F5 / PRE-1 | compiler mutable default | `preprocessing/compiler.py:34` | … |
| 4 | F1 | hardcoded paths in config.yaml | `config.yaml:125,140` | … |
| 5 | F2 | defihacklabs enabled=false | `config.yaml` | … |
| 6 | F-3.1 | defihacklabs.yaml missing confidence_tier T0 | `labeling/crosswalks/` | … |
| 7 | F-3.2 | smartbugs_curated.yaml missing confidence_tier T2 | `labeling/crosswalks/` | … |
| 8 | V-1 | test_patterns.py:7 relative path bug | `tests/test_verification/test_patterns.py:7` | … |
| 9 | V-2 | gate.py:107 flagged_classes asymmetry | `verification/gate.py:107` | … |
| 10 | V-3 | gate.py:143-145 dead code | `verification/gate.py:143-145` | … |

For each row, **open the file at the quoted line, read 20 lines of context, and decide**:

- **FIXED** — code no longer has the issue. Note the new behavior briefly.
- **OPEN** — issue is still present. Note the line where it still exists.
- **DOCUMENTED-DEFER** — code is unchanged but a doc comment + TODO marks it deferred. Acceptable per plan.

**Exit condition:** 10-row table in output doc, every row has a verdict.

### B.2 — Re-verify the 2 CRITICAL items in detail

These two deserve explicit deep-dive, not just a line check:

**B.2.a — REP-2 (class order mismatch):**

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module/sentinel_data && echo "===graph_schema CLASS_NAMES===" && sed -n "70,90p" representation/graph_schema.py && echo "===taxonomy.yaml classes===" && sed -n "20,50p" labeling/schema/taxonomy.yaml'
```

The two orders:
- `graph_schema.py:73-84` (per prior audit): Reentrancy=0, CallToUnknown=1, Timestamp=2, ExternalBug=3, GasException=4, DoS=5, IntegerUO=6, UnusedReturn=7, MishandledException=8, NonVulnerable=9
- `taxonomy.yaml:21-159` (per prior audit): CallToUnknown=0, DoS=1, ExternalBug=2, GasException=3, IntegerUO=4, MishandledException=5, Reentrancy=6, Timestamp=7, TOD=8, UnusedReturn=9

These are STILL divergent. The question is: **is the divergence now documented and the round-trip tested?**

Check for:
- [ ] `representation/README.md` §3 documents the divergence
- [ ] `labeling/schema/README.md` §3 documents the divergence
- [ ] A regression test asserts the v9 checkpoint index order matches `graph_schema.py:73-84` (i.e. Reentrancy=0, NonVulnerable=9)
- [ ] A regression test asserts `taxonomy.yaml` index order matches the merger writes (i.e. CallToUnknown=0, UnusedReturn=9)

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && grep -rn "Reentrancy.*=.*0\|NonVulnerable.*=.*9\|class_names\b" tests/ sentinel_data/ | head -30'
```

**B.2.b — REP-7 (stale_entries wrong sha):**

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && sed -n "85,106p" sentinel_data/representation/cache_manager.py'
```

Read line 95. Per the prior audit: `sha = rep_path.stem` for `abc123.rep.json` returns `abc123.rep` (not `abc123`). The fix is `sha = rep_path.stem.removesuffix(".rep")`.

Verify:
- [ ] Is line 95 fixed?
- [ ] Is there a unit test that catches the bug? (`test_cache_manager.py` or similar)

```bash
wsl -- bash -c 'cd ~/projects/sentinel/data_module && find tests/ -name "*cache*" -o -name "*versioner*"'
```

If no cache test exists, that's `FINDING-B:N` (HIGH).

**Exit condition:** 2 sub-sections in output doc, each with the evidence and verdict.

### B.3 — Walk the P1 list (13 items from `00_EXECUTIVE_SUMMARY.md:165-181`)

Same drill as B.1 — open file, read, decide OPEN/FIXED/DEFER. Output: 13-row table.

### B.4 — Walk the P2 list (14 items from `00_EXECUTIVE_SUMMARY.md:183-198`)

Same drill. Output: 14-row table. P2 items are P1-deferral-eligible; a P2 that's still OPEN gets a "post-Run-11 candidate" tag.

### B.5 — Sample 10 of the P3 items (32 total per `00_EXECUTIVE_SUMMARY.md:200-208`)

P3 is nice-to-have; sample 10 representative items. Output: 10-row table.

### B.6 — Cross-reference with `08_stages_0_2_deep_audit.md`

This doc has its own P0–P3 list (`08_stages_0_2_deep_audit.md:530-598`). Spot-check that the items overlap with `00_EXECUTIVE_SUMMARY.md`. Where they diverge, note the extra items from `08` (they're typically more granular).

Focus areas where `08` adds detail not in `00`:
- `08` lists `PRE-8` (duplicated regex) — verify
- `08` lists `REP-3` (4-levels-up allow_paths heuristic) — verify
- `08` lists `CLI-1` (`_handle_run` drops args) — verify
- `08` lists `TEST-2, TEST-3` (other relative-path tests) — verify

**Exit condition:** additional rows for `08`-only items.

### B.7 — Compute the "compliance score"

Total findings checked: N
- FIXED: X (%)
- OPEN: Y (%)
- DOCUMENTED-DEFER: Z (%)
- UNRESOLVED + CRITICAL/HIGH: W (must be 0 to pass Run 11 gate)

If `W > 0` at end of Phase B, that's a Run 11 blocker — flag it as `FINDING-B:BLOCKER`.

### B.8 — Output doc structure

Author `v2_full_audit/02_phase_b_prior_audit_compliance.md` with:

1. **Executive summary** — compliance score + blocker list
2. **P0 table (10 rows)** — from B.1
3. **CRITICAL deep-dive (REP-2, REP-7)** — from B.2
4. **P1 table (13 rows)** — from B.3
5. **P2 table (14 rows)** — from B.4
6. **P3 sample table (10 rows)** — from B.5
7. **08-only items (variable)** — from B.6
8. **Compliance score** — from B.7
9. **Run 11 blockers** — `FINDING-B:BLOCKER-N`
10. **Open items for Phase D** — items that need end-to-end test to verify, not just file:line check

---

## What this phase will NOT touch

- New findings (those go to Phases C1, C2, D)
- Source code review of files the prior audit didn't touch (Phases C1, C2)
- The seam swap (Phase C2)

---

## Required inputs

- `v2_full_audit/01_phase_a_foundation_recon.md` — Phase A output (for test result + parser state)
- `data_module/audit/00_EXECUTIVE_SUMMARY.md` — primary source
- `data_module/audit/08_stages_0_2_deep_audit.md` — secondary source (more granular)

## Outputs

- `v2_full_audit/02_phase_b_prior_audit_compliance.md` (consumed by Phase E)
- Updated todo list

---

## Exit criteria checklist

- [ ] 10 P0 items checked
- [ ] REP-2 and REP-7 deep-dived
- [ ] 13 P1 items checked
- [ ] 14 P2 items checked
- [ ] 10 P3 items sampled
- [ ] 08-only items cross-referenced
- [ ] Compliance score computed
- [ ] Run 11 blockers identified (must be 0 to pass)
- [ ] Output doc authored with all 10 sections
- [ ] Every row has a verdict (OPEN / FIXED / DEFER)
