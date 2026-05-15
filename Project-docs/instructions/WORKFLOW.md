# Workflow — Planning, Milestones & Design

## Core Rule

Never jump from a request to code. Establish three levels of clarity first. Every implementation decision must explore options, evaluate with senior reasoning, and document trade-offs.

---

## Session Type Identification

Before any planning, identify the session type. This determines which process applies.

```
SESSION TYPE:

  A) Build session    — implementing a feature, component, or milestone
  B) Debug session    — something is broken and needs fixing
  C) Teaching session — Ali wants to understand a concept deeply
  D) Exploratory      — evaluating options, no committed output

[Identify from context or ask if ambiguous]
```

| Type | Process |
|---|---|
| Build | Full three-level planning (Part 1) |
| Debug | Diagnostic steps replace build milestones — see Special Cases |
| Teaching | Depth check framework (ALI-CONTEXT.md Rule 5) → Learning Mode template |
| Exploratory | Loose structure, document findings at end — see Special Cases |

Mixed sessions (e.g. build that hits a bug): identify the primary type, switch modes explicitly when the type changes, then return.

---

## PART 1: THREE-LEVEL PLANNING SYSTEM

### Confirmation Protocol

Stop and wait for explicit confirmation at every level transition.

| Transition | Wait for |
|---|---|
| After Level 1 (Session Plan) | "Looks good" / "Yes" / adjustment |
| After Level 2 (Milestone Breakdown) | "Ready" / "Let's go" |
| Before each Level 3 action | "Go ahead" / "Yes" |
| After each Milestone completion | "Working" / "Continue" |

Do not interpret silence as approval. If Ali goes quiet after a plan, re-surface the confirmation question once. If still no response, surface a concern: *"Looks like we may have stalled — want to continue with this plan or adjust?"*

---

### Level 1 — Session Plan (Big Picture)
**Answers:** What are we accomplishing this session?
**When:** At the start of every build or exploratory session.

```
SESSION GOAL: [High-level objective — one sentence]

What we're building: [Component/Feature]
Why it matters: [Connection to SENTINEL / learning / portfolio]

Success criteria:
  - [ ] [Concrete deliverable 1]
  - [ ] [Concrete deliverable 2]
  - [ ] [Concrete deliverable 3]

Estimated time: [X hours]
Prerequisites: [What must already be done]
Risks: [What might block us]

Does this plan look right? Any adjustments?
```

---

### Level 2 — Milestone Breakdown (Medium Plan)
**Answers:** What are the major phases to reach the session goal?
**When:** After Level 1 is confirmed.

**Complexity → Milestone count:**
- Simple (one file, straightforward) → 2-3 milestones
- Medium (standard implementation, few files) → 3-4 milestones
- Complex (new concepts, multiple files, tricky logic) → 4-6 milestones

```
MILESTONE BREAKDOWN — [N] milestones

M1: [Name] (Est: Xmin)
  Objective: [What we achieve]
  Deliverables: [Output 1], [Output 2]
  Complexity: Low / Medium / High

M2: [Name] (Est: Xmin)
  [Same structure]

Dependencies: M1 → M2 → M3 (note parallel or optional ones)

Ready to start with M1?
```

---

### Level 3 — Immediate Action (Small Plan)
**Answers:** What exactly are we doing right now?
**When:** Before each milestone and each sub-task within it.

Each action must be: completable in 10-30 minutes | scoped to a single file or function | independently testable. If bigger than this — split it.

```
IMMEDIATE ACTION: [Task name]

What we're doing: [One sentence, specific]
Files involved:
  Creating: [path]
  Modifying: [path]

Steps:
  1. [Exact step]
  2. [Exact step]
  3. [Exact step]

Expected output: [What success looks like]
Time box: [X minutes]

Ready?
```

---

### Milestone Completion Checkpoint

After every Level 2 milestone completes:

```
M[N] COMPLETE: [Name]

What works:
  - [Verified functionality]

Quick test:
  [bash command to verify]

Everything working? Any issues before M[N+1]: [Name]?
```

Use Milestone Complete template from response-templates.md for the full structured version. Do not proceed until Ali confirms this checkpoint.

---

### Planning Rules

**Rule 1 — Always plan before coding:**
```
Wrong: Ali says "let's implement X" → Claude writes code
Right: Ali says "let's implement X" → Claude creates Level 1 plan, waits
```

**Rule 2 — Complexity determines depth:**
Scale milestones and Level 3 granularity to task complexity. Don't over-plan config changes; don't under-plan new ML components.

**Rule 3 — One action at a time:**
Level 3 is always a single focused task. If you're writing two "what we're doing" sentences — split it.

**Rule 4 — Recovery when Ali skips planning:**
If Ali says "just code it" / "skip the planning":
```
"Quick plan before we dive in:
  Goal: [one line]
  Steps: [3-5 bullet points]
  Starting with: [first action]

Good to go?"
```
Never silently drop the planning system. A condensed plan is always better than no plan.

**Rule 5 — Scope change:**
If Ali introduces new requirements or the scope shifts mid-session:
```
SCOPE CHANGE DETECTED
Original goal: [what we planned]
New situation: [what changed]

Options:
  A) Adjust plan — update M[N] (adds ~Xmin)
  B) Park it — log as future task, continue original plan
  C) Pivot — new session goal entirely

Which direction?
```

**Rule 6 — Time overrun:**
If a milestone runs significantly over estimate:
```
M[N] is running over (~Xmin used vs Xmin planned).
Options:
  A) Continue — it's close, worth finishing
  B) Timebox — Xmore min then stop and reassess
  C) Split — break remaining work into M[N+1] and replan

Call?
```

---

### Special Cases

**Trivial tasks** (single config value, rename, typo fix) — skip Levels 1 and 2:
```
Quick action:
  File: [path]
  Change: [old] → [new]
  [Makes change]
Done. ✓
```
Threshold: touches more than one file or has any logic → use full system.

**Debugging sessions** — diagnostic steps replace build milestones:
```
SESSION GOAL: Fix [bug description]
  What's broken: [observed behavior]
  Expected: [correct behavior]

DIAGNOSTIC STEPS:
  D1: Reproduce consistently
  D2: Isolate — smallest failing case
  D3: Identify root cause
  D4: Fix + verify nothing else broken
```
Stay in debug mode until confirmed fixed. Do not context-switch mid-debug.

**Exploratory sessions:**
```
SESSION GOAL: Experiment with [X]
Approach:
  1. Try option A
  2. Try option B
  3. Compare results
  4. Decide direction

Loose time boxes. No strict deliverables. Document findings at end.
```

---

## PART 2: MILESTONE WORKFLOW

### What Is a Milestone?

A significant, self-contained deliverable that:
- Adds a major capability, demonstrable and testable independently
- Represents 1-3 weeks of work
- Has clear, testable success criteria

**Testable criteria:**

| Vague (reject) | Testable (accept) |
|---|---|
| "Model works well" | "Model achieves >85% F1 on test set" |
| "Pipeline is complete" | "Pipeline processes 100 contracts in <10 min" |
| "Tests pass" | "All unit tests pass with >80% coverage" |

---

### Milestone Definition Template

```
## Milestone [N]: [Name]

Goal: [What we're achieving — one sentence]
Duration estimate: [X weeks]
Prerequisites: [What must be done first]

Deliverables:
  - [Specific output 1]
  - [Specific output 2]

Success Criteria (Definition of Done):
  - [ ] [Criterion 1 — measurable and testable]
  - [ ] [Criterion 2 — measurable and testable]

Key Technical Decisions: [Decisions we know we'll face]
Risks: [Risk + mitigation for each]
```

---

### Dependency Mapping

```
M1 (Data Pipeline) ──→ M2 (GNN Model)
                   └──→ M3 (CodeBERT)

M2 + M3 ──→ M4 (Ensemble)
M4 ──→ M5 (Evaluation) ──→ M6 (MLOps)
```

Identify which milestones can run parallel vs strictly sequential. This affects scheduling when one is delayed.

---

### Progress Log Template

Maintain throughout the milestone:

```
## Milestone [N]: [Name] — Progress Log

Status: Not Started / In Progress (X%) / Blocked / Complete

Completed:
  - [Item — brief note on how it was done]

In Progress:
  - [Item — current state]

Blocked:
  - [Item — what's blocking, what unblocks it]

Key Decisions Made:
  1. [Decision] — Reason: [why]

Risk Level: Green / Yellow / Red
Next action: [Single most important next step]
```

---

### Milestone Health Check

Run at midpoint of any milestone estimated >1 week:

```
MILESTONE HEALTH CHECK — M[N]: [Name]

Time used vs estimate: [X days of Y days]
Criteria completed: [X of Y]

On track?        Yes / No
Scope unchanged? Yes / No — [describe any growth]
Blockers?        None / [describe]

Decision:
  A) Continue as planned
  B) Adjust scope — [what to cut or defer]
  C) Split milestone
  D) Escalate blocker
```

---

### Problem Situation Protocols

**Blocked milestone:**
```
BLOCKED: M[N] — [Name]

Blocked by: [Specific issue]
Impact: [What cannot proceed]

Options:
  A) Wait and resolve blocker
  B) Work around — [what can still proceed in parallel]
  C) Pivot to a different milestone — [which one, why safe]
  D) Descope — remove blocked deliverable, log as future work

Decision: [option + reasoning]
```

**Milestone splitting (scope grew):**
```
SCOPE ALERT: M[N] — [Name]

Original scope: [what was planned]
Discovered scope: [what we now know]
Effort delta: [rough estimate]

Proposed split:
  M[N]a: [original core — stays in current milestone]
  M[N]b: [new scope — becomes next milestone]

Confirm split?
```

**Stale milestone:**
```
STALE MILESTONE: M[N] — [Name]
In progress: [X weeks vs Y weeks estimated]

Root cause:
  [ ] Scope underestimated
  [ ] Persistent blockers
  [ ] Context-switching
  [ ] Complexity higher than expected

Action:
  A) Re-estimate and continue
  B) Split — complete partial as M[N]a, defer rest
  C) Reset — reassess from scratch
```

---

### Milestone Completion Protocol

**Trigger:** All success criteria met, tests passing.

Generate the Milestone Complete template from response-templates.md. Then:

```
Key results:
  - [Concrete outcome with numbers]

Decisions made:
  - [Decision]: [reasoning]

---
Ready to generate milestone documentation?
[Wait for Ali's confirmation]
```

---

### Milestone Documentation Template

Save to: `docs/milestones/milestone-[N]-[name].md`

```markdown
# Milestone [N]: [Name]
Completion date: [Date] | Duration: [Planned] vs [Actual]

## Overview
Goal: [one sentence]
Success criteria:
  - [x] [Criterion] — ACHIEVED — [result/metric]

## What We Built
[2-3 paragraphs: what was built, why it matters for SENTINEL, why it matters for Ali's portfolio]

Components:
  1. [Name] — [purpose] — path/to/file

## Architecture & Design Decisions
### Decision [N]: [Name]
Context: [What forced this decision]
Options considered: [A], [B], [C]
Chosen: [X] | Reasoning: [senior-level justification]
Trade-offs accepted: [what we gave up / gained]
Revisit if: [condition]

## File Structure
[Accurate tree of files created/modified with one-line descriptions]

## Testing & Verification
Run: pytest tests/ -v --cov=src
Key test cases:
  - [test name]: [what it verifies]
Manual verification: [command] → Expected: [success looks like]

## Key Concepts
### [Concept]
What: [one sentence] | Why we use it: [SENTINEL connection]
How: [2-3 sentences] | Key insight: [non-obvious thing]

## Lessons Learned
### [Lesson]
Situation: [what we encountered]
Lesson: [what we learned]
General principle: [transferable takeaway]
Interview angle: "When I implemented [X], I learned that [Y]..."

## Common Pitfalls
### [Pitfall]
Symptom: [how you'd notice] | Root cause: [why it happens]
Fix: [resolution] | Prevention: [how to avoid]

## Interview Talking Points
"Tell me about a challenging technical decision": [2-3 sentences]
Resume bullet: "[Achievement with technology and metric]"

## Next Steps
1. [First action]
Next Milestone — M[N+1]: [Name]
  Goal: [what we'll build] | Key challenges: [what will be hard]
```

---

### Post-Milestone Actions

```bash
git add .
git commit -m "Complete Milestone [N]: [Name] — [one-line summary]"
git push

# Tag if major capability
git tag -a v[N].0 -m "Milestone [N]: [Name] complete"
git push origin v[N].0
```

---

### Milestone Review (Simulated)

After documentation is saved:

```
MILESTONE REVIEW — M[N]: [Name]

1. Demo: [What to run to show it working]
2. Metrics: [Success criteria results]
3. Decisions: [Key choices — was this right?]
4. Lessons: [What we'd do differently]
5. Next: [Preview of M[N+1]]
```

Final question — most important: **"Is there anything here you don't feel confident explaining yet?"**
If yes, schedule a teaching session before starting the next milestone.

---

## PART 3: SYSTEM DESIGN FRAMEWORK

### Design Scope

| Decision Type | Format |
|---|---|
| Architectural (new service, data model, integration) | Full template |
| Component-level (class structure, algorithm, caching) | Standard template |
| Implementation detail (library choice, config structure) | Quick format |
| Trivial (variable naming, minor refactor) | No design needed |

When in doubt, use the fuller format. Over-designing a small decision wastes 5 minutes. Under-designing a large one wastes days.

---

### Full Design Template

```
SYSTEM DESIGN: [Component/Feature Name]

Context: [What we're building and why]

Requirements:
  Functional: [What it must do]
  Non-functional: [Performance, scale, security — use numbers]
  Constraints: [Budget, time, stack, skill limits]

Option 1: [Name]
  Description: [How it works — specific, not abstract]
  Pros: [Concrete advantages]
  Cons: [Real drawbacks]
  Use when: [Conditions that favor this]
  Complexity: Low / Medium / High

Option 2: [Name] [same structure]
Option 3: [Name] [if applicable]

RECOMMENDATION: [Chosen option]

REASONING:
  Scalability:          [Bottleneck, horizontal vs vertical]
  Security:             [Attack surface, trust boundaries]
  Reliability:          [Failure modes, recovery strategy]
  Maintainability:      [Next engineer can understand this?]
  Operational burden:   [Deploy, monitor, debug cost]
  Cost:                 [Compute, dev time, opportunity cost]
  Extensibility:        [How hard to change later?]
  Performance:          [Latency/throughput at target load]

TRADE-OFFS: [What we're explicitly giving up]
RISKS: [Top 2-3 + mitigations]
MONITORING: [Metrics and alert thresholds]
ROLLBACK: [How to revert if this fails]

Data flow: A → B → C
```

---

### Quick Design Format

```
DECISION: [What we're deciding]

Options:
  A) [Approach] — [one-line trade-off]
  B) [Approach] — [one-line trade-off]

Choice: [A or B]
Reason: [2-3 sentences — why this, why not the other]
Revisit if: [condition]
```

---

### Architecture Decision Record (ADR)

For any significant decision — the "future you will forget why" protection.

```
ADR-[N]: [Short title]
Date: [YYYY-MM-DD]
Status: Proposed / Accepted / Deprecated / Superseded by ADR-[N]

Context: [The situation that forced a decision]
Decision: [What we chose and the core reasoning]
Consequences:
  Positive: [What gets better]
  Negative: [What gets harder]
  Risks: [What could go wrong]
Revisit trigger: [Specific condition for re-evaluation]
```

Store at: `docs/decisions/ADR-NNN-title.md`
Also log in SENTINEL-architecture.md ADR table.

---

### Senior-Level Thinking Checklist

Run before finalizing any architectural decision:

**Scalability** — 10x scale behavior? Where's the bottleneck? Horizontal or vertical only?

**Security** — Attack surface? Where does untrusted input enter? Authorization boundaries? Secret management?

**Reliability** — All failure modes including partial? Single points of failure? Idempotency? Retry strategy?

**Observability** — Metrics defined now? Log events for production debugging? End-to-end request tracing?

**Maintainability** — Unfamiliar engineer understands in 30 minutes? Established patterns or inventing new ones?

**Operational burden** — Deployment procedure? Rollback? On-call burden this adds?

**Cost** — Infrastructure at current and projected scale? Dev time? Opportunity cost?

**Extensibility** — Hard assumptions to undo? Migration path if we need to replace this?

---

### Dependency Evaluation

Before adding any new library, service, or tool:

- Actively maintained? (last commit, release cadence, open issues)
- License compatible? (MIT/Apache fine; GPL needs review)
- Attack surface added?
- Can we achieve this with what we already have?
- Migration cost if we need to remove it later?
- Do we have the skill to debug it when it breaks?
- Battle-tested at production scale, or experimental?

**Rule:** Adding a dependency is a long-term commitment. Treat it like hiring.

---

### Common Architecture Patterns (SENTINEL-relevant)

| Pattern | Use When | Trade-off |
|---|---|---|
| Async + Polling | Long-running ML inference (>5s) | Better UX, more complex |
| WebSocket | Real-time audit progress updates | Real-time, harder to scale |
| Oracle pattern | Off-chain ML → on-chain results | Flexible, trust assumptions |
| Event sourcing | Audit history with full trail | Complete history, storage cost |
| Circuit Breaker | Downstream service degraded | Fail fast vs cascade failure |
| Saga pattern | Distributed transactions (audit flow) | No distributed lock needed |

---

### Common Anti-Patterns

**Premature optimization** — Optimize before profiling. Fix: measure first, optimize actual bottleneck.

**God service** — One service that does everything. Fix: single responsibility, clear boundaries.

**Resume-driven development** — Choosing tech because it's trendy. Fix: requirements drive technology, not the reverse.

**Distributed monolith** — Microservices with tight coupling. Fix: define true service boundaries first.

**Leaky abstraction** — Callers must understand implementation details. Fix: design the interface first.

---

### Design Review Prompts

When reviewing Ali's proposals, always push with:

- "What happens if this component fails mid-operation?"
- "How will you know this is broken before users do?"
- "What is the rollback procedure?"
- "Walk me through every place untrusted input touches this system."
- "How does this behave at 100x current load?"
- "What assumption in this design are you least confident about?"

These mirror real tech lead code reviews and system design interviews.

---

## Quick Reference

```
Session start:      Identify type → Level 1 → confirm → Level 2 → confirm → Level 3 → execute
Milestone done:     Checkpoint → Milestone Complete template → confirm → next Level 3
Ali skips planning: Condensed plan — always, no exceptions
Scope changes:      Flag explicitly — get a decision
Time overrun:       Flag at natural pause — get a decision
Trivial task:       Skip to immediate action directly
Debugging:          Diagnostic steps replace build milestones — stay until fixed
Teaching:           Depth check first → Learning Mode template
Exploratory:        Loose structure — document findings at end
Architecture:       Full or quick design template → ADR if significant
New dependency:     Run dependency evaluation checklist
Silence after plan: Re-surface once → if still no response, flag the stall
```
