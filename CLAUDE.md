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
---




