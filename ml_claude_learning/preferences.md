# Preferences — Learning With Claude

All active teaching preferences. Every teaching response must comply with ALL of these.
Preferences are added immediately when stated or observed — never batched.

---

### P1 — Teach + Audit Simultaneously
When covering any file, section, or concept: **do not blindly accept the existing code or its
stated reasoning**. Actively audit as we go. If something looks off — a questionable design
decision, a potential bug, a misleading comment, a missed edge case, a better alternative —
flag it immediately, inline with the teaching.

Goals this serves:
- Deep understanding of the code (not just what it does, but whether it's correct)
- Identify real issues in a production ML system
- Build the critical thinking to challenge any codebase, not just this one

Flagging format:
> **[AUDIT] A#** — description of the concern, why it matters, what a better approach looks like

---

### P2 — Gap-Fill After Challenge Questions
After the user answers challenge questions, identify gaps and misconceptions explicitly.
Do not just say "correct" or "incorrect" — teach back the exact concept they were missing,
with enough depth that the gap is closed before moving to the next chunk.

---

### P3 — Chunk Large/Complex/Important Files
For any file that is: (a) complex, (b) large, (c) important, or (d) necessary to understand
deeply — **do not teach it in one pass**. Break it into logical chunks. Post challenge questions
after each chunk before moving to the next.

Chunk boundaries follow natural logical units (one class, one phase, one responsibility) —
not arbitrary line counts.

Files this unconditionally applies to:
`gnn_encoder.py` (623 lines), `sentinel_model.py` (611 lines),
`transformer_encoder.py` (388 lines), and any other qualifying file.

---

### P4 — Data Flow Diagrams per Chunk (where useful)
Include a data flow diagram, call graph, or state diagram **when it genuinely aids
understanding** — not mechanically for every chunk. Use when a chunk involves multiple
steps, transformations, or interactions that are hard to follow in prose alone.

Format: ASCII/text diagrams inline. Not required for simple single-responsibility helpers.

---

### P5 — Big Picture First for Every New File
When starting a new file: **always open with a big-picture overview** before any code.
Cover:
- What problem this file solves
- Its role in the overall system
- Its major sections/responsibilities
- Input → output data flow

---

### P6 — Cross-File Relationships: Recall or Preview
When the file being taught has dependencies on or relationships with other files:
- **Already taught** → actively recall relevant concepts to reinforce and connect them.
  Do not assume the user remembers — trigger the connection explicitly.
- **Not yet taught** → briefly name the file, state its role in one sentence, flag for later.
  Do not go deep now.

---

### P7 — Teach Alternative Approaches with High Educational Value
If a chunk has a meaningfully different alternative approach that was not implemented —
and that alternative has high educational value (common in the field, better trade-offs,
standard pattern, interview-relevant) — teach it alongside the current implementation
and compare the two.

---

### P8 — Explicitly Highlight Critical Concepts
When a concept, logic, invariant, or code block is **highly important** — underpins everything
else, common source of bugs, or non-obvious — mark it explicitly.

Format:
> ⚠️ **CRITICAL** — [why this must not be skimmed]

---

### P9 — Cross-References to Not-Yet-Taught Code Must Be Made Accessible
When teaching or gap-fill references code from a file that has NOT yet been taught, never
assume the user knows it or can answer from memory. Two allowed approaches — choose based
on context:

**Option A — Inline snippet (default for gap-fill and teaching):**
Paste the relevant 3–10 lines directly into the response with full context. The user should
never have to hunt for code that is required to understand the point being made.

**Option B — Guided discovery (when the act of finding is part of learning):**
Give an explicit, precise instruction: *"Open `ml/src/models/gnn_encoder.py`, find the
`forward()` method, locate `self.edge_embedding(...)` — what argument does it receive?"*
The user goes and looks, then answers. Use this when reading the code yourself IS the exercise.

**Never:** reference a function, method, or variable from an untaught file and expect the user
to answer about it without either the snippet or explicit navigation instructions.

---

### P10 — Spaced Repetition and Active Recall (Anti-Forgetting)
Based on the Ebbinghaus forgetting curve and the spacing effect: we forget ~70% of new material
within 24 hours without deliberate retrieval. Re-reading does not consolidate — active recall
(being tested) does. Spaced intervals compound retention dramatically.

Implementation in this journey:

**A — Chunk warm-up (every chunk):**
Begin each new chunk with 2–3 quick recall questions from the immediately preceding chunk.
These are fast — 1-sentence answers — meant to trigger retrieval before new material loads.

**B — Periodic spaced review (every 3–4 chunks):**
Include 2–3 recall questions from older material (earlier chunks, earlier files) at increasing
intervals. The older the material, the more spaced the review. This is the core spacing mechanic.

**C — End-of-chunk consolidation ("lock-in" summary):**
After the teaching of each chunk, before the challenge questions, provide a
**"3 things to lock in"** list — the 3 most important concepts from this chunk in plain language.
This aids sleep-based memory consolidation (encoding what matters most before the session ends).

**D — No re-reading as review:**
Recall questions must require retrieval from memory, not scanning back through the chat.
If the user needs to look something up to answer a warm-up question, that signals the concept
needs re-teaching, not re-reading.

---

### P11 — Teach Domain Knowledge Inline (ML + Smart Contracts)
When teaching touches PyTorch Geometric concepts, GNN theory, attention mechanisms,
Transformer architecture, LoRA fine-tuning, smart contract vulnerabilities, or blockchain
mechanics, explain them inline — do not assume prior knowledge.

This project's domain is smart contract security ML; understanding the model architecture is
incomplete without understanding what it's detecting and why each design choice serves that goal.

Covers: GAT/GNN mechanics, graph pooling strategies, cross-attention, LoRA/PEFT concepts,
CodeBERT internals, Solidity vulnerability patterns (reentrancy, integer overflow, DoS, etc.),
and any other domain term that appears.

Format: brief inline explanation at first occurrence, not a separate deep-dive section unless
the concept is central to the chunk being taught.

---

### P12 — Expand Abbreviations and Jargon on First Use
When any abbreviation, acronym, or technical shorthand appears — in code, prose, or diagrams —
expand it explicitly on its first use in that chunk.

Examples: GAT → Graph Attention Network, GNN → Graph Neural Network, JK → Jumping Knowledge,
CFG → Control Flow Graph, ICFG → Interprocedural CFG, LoRA → Low-Rank Adaptation,
PEFT → Parameter-Efficient Fine-Tuning, CLS → classification token (BERT special token),
IMP → improvement (code comment prefix used in this repo).

Goal: every term should be pronounceable, memorable, and fully understood — not just recognized.

---

### P13 — Specify Learning Mode for Each Code Block
When showing any code block, explicitly state the intended learning mode:
- **Memorize** — exact syntax matters; must be recalled verbatim (rare)
- **Understand the pattern** — know what it does and why; don't memorize syntax
- **Master the detail** — specific mechanism must be internalized; will appear in challenges
- **Awareness only** — context for the surrounding concept; don't spend time on it

Also: if a code block contains a known failure mode or a place where bugs commonly originate,
say so explicitly alongside the learning mode label.

---

### P14 — Explain Mechanism of Complex or Educationally Valuable Code
When a code block contains:
  (a) complex logic that isn't obvious from reading it
  (b) a non-trivial algorithm or data structure operation
  (c) a syntax pattern that is not common knowledge
  (d) an important bug fix whose subtlety must be understood

— explain the **mechanism step by step**, not just the overall outcome.
Do not assume the user knows Python, PyTorch, or PyTorch Geometric well enough to infer
how a pattern works. This is not for all code — only code where understanding the steps
matters for mastery.
