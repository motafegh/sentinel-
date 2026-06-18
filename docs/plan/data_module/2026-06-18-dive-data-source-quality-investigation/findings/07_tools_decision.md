# Method 7 — Tools Available + Add Decision

**Status:** COMPLETE — 2026-06-19

---

## Verified inventory

| Tool | Version | Location | Status | Phase 1 utility |
|---|---|---|---|---|
| Slither | 0.11.5 | `.venv/bin/slither` | ✅ Working, documented | RE: detects CEI (reentrancy-eth/no-eth). EB: no detector. |
| Aderyn | 0.6.8 | `~/.cargo/bin/aderyn` | ✅ Working, documented | RE: constructor CEI only (public/external). EB: no detector. |
| Echidna | 2.3.2 | `~/.local/bin/echidna` | ✅ Installed, tested | RE+EB: zero signal without custom assertions. Not useful. |

### References

- `docs/plan/data_module/slither_reference.md`
- `docs/plan/data_module/aderyn_reference.md`
- `docs/plan/data_module/echidna_reference.md`

---

## Decision: No additional tools to install

| Candidate | Independent signal? | Verdict |
|---|---|---|
| Echidna 2.3.2 | ❌ Requires assertions DIVE contracts don't have | Installed but not for Phase 1 |
| Mythril | ⚠️ Symbolic execution — but slow | Not for corpus-scale use. Revisit for targeted analysis. |
| Semgrep | ❌ Similar static pattern matching to Slither | Redundant. |
| Manticore | ⚠️ Symbolic execution — very slow | Not for corpus-scale use. |

**Rationale:** Both Slither and Aderyn are working and documented. Neither detects missing access control (EB) — this is a fundamental blind spot in available static analysis tools, not a missing-tool gap. Adding more static tools would not fill this gap because they share the same pattern-matching paradigm. For Phase 1 (EB/RE only), the current tool set is sufficient for providing investigative hints during manual review.

**When to revisit:** Phase 2 (full DIVE — remaining classes like DoS, Timestamp, IntegerUO) may benefit from Echidna for contracts with assertions, or Mythril for targeted symbolic execution on narrowed sets. Phase 3 (other sources — BCCC, SolidiFI) may require additional tools depending on the source's contract characteristics.
