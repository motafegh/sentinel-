## Who Ali Is

Ali —  learining by doing  SENTINEL project ( decentralised smart contract security oracle).
Current target: Senior AI/ML Engineer and Hybrid AI/Blockchain Engineer roles.

| Environment | Windows 11 + WSL2 Ubuntu + VS Code | Use Linux paths inside WSL2 | RTX 3070 8GB VRAM, cpu:i7-12700H, 64GB RAM | Python 3.12.1, Poetry, venv |

Current project: SENTINEL at `~/projects/sentinel`.
**SENTINEL project state, training history, schema, key file paths, audit findings, etc.**
**live in:** `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` (and the
sibling `project_*.md` files in that directory). Read MEMORY.md at session start.



---

## Rules

**0 - Always answer Honestly, Professionally and in production ready ,educational valuable learning lens**

**1 — Plan before code.**
Write a markdown file confirming the session plan before any implementation. If Ali says "just code it" — surface a
condensed plan, wait for a nod (yes / go / looks good), then proceed. Continued questions
are not a nod.

**2 — Check MEMORY.md.**
Before any architecture or library decision, check the live `MEMORY.md` and all of its
references that you need for your works at:
wsl path : 
`/home/motafeq/.claude/projects/-home-motafeq-projects-sentinel/memory/`

windows path : 
`\\wsl.localhost\\Ubuntu-24.04\\home\\motafeq\\.claude\\projects\\-home-motafeq-projects-sentinel\\memory`

**3 — WORKING MEMORIES : Document incrementally, never batch.**
Whenever planning, analyzing, or reading source code:
- Before starting, create a dedicated `.md` file at a path like
  `~/.claude/scratch/<topic>_<YYYYMMDD>.md` (e.g. `gnn_trainer_analysis_20260613.md`).
- Write findings to that file **continuously and incrementally** — after every hypothesis
  formed, every ambiguity spotted. Do NOT hold findings in context and flush later.
- The file is the working memory. If the session ends or context resets, nothing is lost.
- Update the file even for small steps: a one-line note beats a lost insight.
- When the session goal is complete, summarize key findings back into MEMORY.md
  (keep MEMORY.md ≤ 200 lines; link to the scratch file if needed).

**4 — Trust source code only. Distrust all docs.**
When reading the SENTINEL codebase:
- **Canonical truth**: `.py`, `.sol`, `.ts`, `.sh` files — the actual executable syntax.
- **Explicitly untrusted** (treat as stale until verified against code):
  - Docstrings and inline comments (they lag behind refactors)
  - README files, any `.md` docs, `Project-docs/`, `docs/`
  - Function signatures in documentation that differ from actual `.py` files
- Workflow: read the source first → form your own understanding → only then cross-check
  docs if curious. Never infer behavior from a docstring without confirming in the
  function body.
- If a docstring contradicts what the code does: **the code is right**. Flag the stale
  docstring as a finding in the scratch `.md` file.

**5 — Professional coding standards** (always on, all modules ,submodules and everywhere).

When writing or refactoring: apply by default.
When reviewing: flag violations as findings in the scratch `.md` file.

**A — Single Responsibility.**
One file, one reason to change. If describing what a file does requires "and", it should be two files.

Heuristics (guidelines, not laws): functions ~50 lines; files 200–400 lines; past 500 needs
a reason; past 1000 split it.

Prefer a package of focused modules (`nodes/<one_per_node>.py`) over a god-file
(`nodes.py` with 13 responsibilities). Appending to the current file is the AI default —
default to splitting instead.

**B — No decision-number changes without measurement.**
Any threshold, weight, or confidence value is POLICY. It must be:
(a) externalized into versioned config — never a constant buried in `.py`
(b) changed only when a measurement justifies it

Tests prove the code runs. Evals prove the system is good. A passing suite is necessary,
not sufficient.

If Ali proposes changing a decision number, ask for the before/after eval result before
proceeding. "I think 0.35 feels right" is not sufficient. "0.35 gives recall 0.91 /
precision 0.62 on the held-out set" is.

Maturity ladder — every decision number has a level:
- Level 0: hand-set constant (prototype only — label it as such)
- Level 1: externalized versioned config
- Level 2: measured against a baseline before every change
- Level 3: learned from data (e.g. threshold from precision-recall curve; weight fitted
  from per-tool confusion matrix)


for learning artifacts explicitly ask Ali where to store them (e.g. `~/projects/sentinel/learning_artifacts/`),no need to say about them in MEMORY.md. 

**C — No silent failures. No silent skips. No "fail-soft that hides the failure."**

While we are building and trying different things, the system must never swallow a failure and
continue as if nothing happened. A silent skip is a lie told to the next layer, which then
computes a number against an absent witness (the Aderyn bug: `FileNotFoundError` caught,
logged at `DEBUG`, `[]` returned indistinguishable from "ran clean" → 83 contracts of zero
Aderyn findings → biased reliability matrix → biased macro-F1). **A silent failure
manufactures a rabbit hole.**

Applies to (non-exhaustive): subprocess tool calls (slither, aderyn, halmos, gigahorse,
ityfuzz, anvil), MCP tool calls, external HTTP, optional dependency imports, fallback paths,
default-return-on-exception, env-var substituters, "tool not installed" branches, and any
`except Exception: return []` / `return {}` / `return None` pattern.

**The contract — any of these is acceptable, none of them is silent:**
1. **Eager error** — `raise` with a precise message naming the tool, the resolved path
   (or `shutil.which` == None), and the exact precondition that failed.
2. **Structured degraded return** — return a value that **carries the failure**, not one
   that hides it: a finding dict with `detail={"skipped": True, "reason": …, "resolved": …,
   "exit": …}` OR a sentinel whose `skipped` flag downstream code MUST check. Audit report
   MUST serialise this so the eval layer sees it.
3. **Explicit node-status field** — `state["tool_status"][tool] = {"ran": False, "reason": …}`,
   and synthesizer / eval MUST treat absent `ran=True` consistently. An empty return is no
   longer "tool absent"; `ran=False` is.

**NOT acceptable:**
- `except FileNotFoundError: logger.debug("not installed — skipping"); return []`
- `except Exception: return [] or {}` with no status mutation
- Logging only (log lines drop at thresholds, gone by report time)
- A debug log + empty return that, from the caller's view, is identical to "ran clean"

**Migration:** apply to new code immediately. For existing silent-skip sites found during
other work, stop and record a `file:line` finding in the scratch `.md` (what fails silently,
what downstream measurement it contaminates, intentional-or-not), but do NOT refactor all in
one sweep — surface to Ali one at a time. The Aderyn site (`_helpers.py:80-118`) is the
first finding.

**Tests:** a unit test mocking a subprocess to raise `FileNotFoundError` MUST assert the
code surfaces a status field or raises — it must NOT assert `result == []` against a value
indistinguishable from "ran clean." Audit the existing tests when a silent-skip site is
refactored.




