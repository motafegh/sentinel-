# Audit Flags — Learning With Claude

All issues found during teaching. Every entry is permanent — never delete, only append.
Entries are added **immediately** when an `[AUDIT]` flag is raised inline during teaching.

Format per entry:
```
## A# — File — Short description
**File:** path
**Location:** function/line
**Issue:** what is wrong and why it matters
**Fix:** concrete fix
**Severity:** Low / Medium / High
**Status:** Open / Noted / Fixed
**Raised:** Session N, Chunk N
```

Severity scale:
- **High** — correctness bug, data loss, silent wrong result, security hole
- **Medium** — design flaw, missing guard, misleading abstraction, perf trap
- **Low** — style issue, suboptimal naming, missed documentation, minor inefficiency

---

<!-- No audit flags yet. First entry will be A1, raised during Session 1. -->
