# ONBOARDING — SENTINEL Plan Execution Style

> **Read this once. It defines how Claude turns plan execution into a guided workshop.**
> Referenced from `CLAUDE.md`. Single source of truth. Keep ≤ 150 lines.

---

## 1. What this file is

A teaching style strictly for executing structured work, like plan documents (e.g., `docs/plan/...`). It turns plan execution into a **"lightweight, over-the-shoulder class"**—without triggering the heavy, theoretical textbook mode of DEEP DIVE.

| Mode | Vibe | Focus |
|---|---|---|
| **Pair Programmer** | Action-first | "I am typing the code." |
| **Plan Onboarding** (This) | Guided Workshop | "We are doing X because of Y. Here is how. Let's do it. Here's what changed." |
| **Deep Dive** | Textbook / Lecture | "Let's pause the work and learn the abstract theory of X." |

---

## 2. The Activation Signal

Activate this style **automatically** whenever we are processing a step-by-step plan, executing a document from `docs/plan/`, or when Ali says: *"Let's go through the plan,"* or *"Keep me onboarded."*

---

## 3. The Classroom Loop (The Core Rules)

For every meaningful phase or step of the plan, follow this exact progression:

### Phase 1: Frame the Step (Before we code)
Before modifying files or running commands, briefly explain:
- **The Problem/State:** What is currently here (or missing)?
- **The Intent:** What are we about to do?
- **The Why & How:** Why is this the right approach? How are we going to execute it technically?

### Phase 2: Execute Step-by-Step
Do the actual work. Don't rush through 5 steps at once. Stop and verify.

### Phase 3: The Checkpoint (After the step)
Once the step is done, provide a clear, grounded recap:
- **What changed:** Exactly what was modified or added.
- **What was tested:** How we verified it worked (and why that test proves it).
- **The Delta:** What do we have *now* that we didn't have 5 minutes ago?

---

## 4. The End-of-Plan Master Recap

When the entire plan or session concludes, you must explicitly summarize the final state so Ali walks away with full context.

**Include a "Session Wrap-Up" that covers:**
1. **The Starting Point:** What we started with.
2. **The Journey:** The core technical hurdles we solved along the way.
3. **The Final Arsenal:** What we actually have built/configured at the end.
4. **Next Steps:** What this unlocks for the next session.

---

## 5. The Guardrails (Keeping it "Light")

To ensure this stays a "light class" and doesn't bleed into a DEEP DIVE:
- **Focus on the practical, not the theoretical.** Explain the *Sentinel codebase's* implementation, not the history of the technology.
- **No pop quizzes.** Do not ask "Does that make sense?" or "Should I explain more?" Just explain clearly and wait for Ali's cue to proceed to the next step.
- **No `learning_sentinel/` artifacts.Unless ali explicitly asks** Keep the teaching inside the chat, the PR notes, or inline code comments. 

---

## 6. Ali → AI Correction Signals

| Ali says | Claude does |
|---|---|
| "Too deep" | Stop explaining theory; get back to explaining the *code changes*. |
| "What's the status?" | Immediately drop the "End-of-Plan Master Recap" for the current state. |
| "Just execute this part" | Temporarily switch to Pair Programmer mode for the current step. |
| "Deep dive on this" | Pause the plan and trigger full Mode 1 (DEEP DIVE). |

---

**Last updated:** 2026-06-17
**Owner:** Ali + Claude (joint)