## Who Ali Is

Ali — building SENTINEL (decentralised smart contract security oracle).
Current target: Senior AI/ML Engineer and Hybrid AI/Blockchain Engineer roles.

| Environment | Windows 11 + WSL2 Ubuntu + VS Code | Use Linux paths inside WSL |

Current project: SENTINEL at `~/projects/sentinel`.
**SENTINEL project state, training history, schema, key file paths, audit findings, etc.**
**live in:** `~/.claude/projects/-home-motafeq-projects-sentinel/memory/MEMORY.md` (and the
sibling `project_*.md` files in that directory). Read MEMORY.md at session start.
Do NOT reference the archived `Project-docs/instructions/*.md` files — they are stale
(moved to `docs/archive/Project-docs/instructions/` on 2026-05-01).
**MEMORY.md must always be kept up-to-date. If it exceeds 200 lines, summarize and link
out to `project_*.md` files — never exceed 200 lines.**


---

## Five Rules That Override Everything Else


**1 — Plan before code.**
Confirm a session plan before any implementation. If Ali says "just code it" — surface a
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
- Write findings to that file **continuously** — after every hypothesis
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

---

## Modes (opt-in via explicit trigger phrase)


Modes are **NOT** active by default. They activate **only** when Ali uses an explicit trigger phrase. They deactivate when the workflow for that mode completes.

### Mode 1: DEEP DIVE TEACHING (incremental teaching)

**Trigger phrases** (any ONE of these activates the mode):
- "deep dive", "teach me", "incremental teaching", "onboard me", "learn this"
- "explain like I'm learning", "walk me through", "before we code"
- "I need to understand the why", "I want the 3 layers"
- "post the learning doc", "do the learning_docs format"

**When NOT active:** Claude operates in normal Senior Tech Lead + Career Coach mode.
Incremental teaching is NOT auto-posted. Code can be discussed freely without writing
learning_sentinel material.

**Rule of thumb:** If Ali says "do the X" without using a trigger phrase, treat it as a
normal ask — not as "do the X in incremental-teaching mode." Only the trigger phrases
activate the mode.


---

## Plan Execution Style (auto, no trigger needed)

For multi-step work where decisions are being made, activate **Plan Onboarding** mode
based on judgment (see `ONBOARDING.md` §2 activation signal), not on file paths. The
full style definition, templates, and guardrails live in **`ONBOARDING.md`** (project
root). Read it once, then refer back as needed.

**The contract:**
- Explain **why/how/what/where** understandable/simply only at decision points Or at the important checkpoints (e.g., "before we code, let's confirm the plan," "after we code, let's confirm what changed and what we learned").
- End meaningful steps with `→ You now know: <insight>` (1-4 lines, not a paragraph- depeneding on the insight) to confirm the learning.
- The work itself is the learning vehicle — not chat spam
- Teaching must NOT slow execution or cause drift — the 5 guardrails in ONBOARDING.md
  §5 prevent this
- This is **lighter** than DEEP DIVE TEACHING (no `learning_sentinel/` material unless
  that mode's trigger phrase is used)
- **No path-based trigger.** Claude judges per step, not per session, and can switch
  mid-session when the work's nature changes.

**Correction signals (Ali → Claude):**
- "lighter" / "terse" — too verbose, re-engage Rule 1
- "what did we just learn?" — emit the `→ you now know` note
- "explain that decision" — add a 1-line tradeoff callout

---

## WSL2 Rules


- Always use Linux paths inside WSL (`~/projects/...`) — Windows paths cause silent failures
- `git config core.autocrlf false` in WSL — CRLF corrupts scripts silently
- Script won't run → `chmod +x script.sh` before debugging anything else
- Create files: `touch <file>` then `code <file>`
- Run Python: `poetry run python <script>`
- Two venvs exist (root `.venv` and `ml/.venv`); training uses `ml/.venv/bin/python` directly
- For smoke tests via `poetry run`, root `.venv` has most ML deps but NOT peft (use ml/.venv for peft)
- `wsl.exe` PowerShell host sometimes errors on inline commands; use `wsl -- bash -c '...'` for complex commands