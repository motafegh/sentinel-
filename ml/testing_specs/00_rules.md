# 00 — Universal Rules

> These rules apply to **every procedure in every spec file** in this folder.
> They are not repeated in individual files. They are always in force.
> Load this file alongside whichever section spec you are using.

---

## Rule 0 — Read Before Claiming

Never assert a value, state, or result without first reading its source.

- Do not use memory, prior conversation, or assumption as the source of a fact
- Do not re-read a file and then quote it from memory — cite which file and section
- If the file does not exist yet, say so explicitly — do not estimate its content
- If the file exists but is ambiguous, resolve the ambiguity before proceeding

This rule applies to: schema constants, checkpoint paths, metric numbers, run names,
label counts, known bugs, config values, field names, and architectural decisions.

---

## Rule 1 — Where Things Live

Read from the correct source. Do not look for values anywhere else.

| What you need | Where to read it |
|---|---|
| Schema constants (dims, types, edge types) | `ml/src/preprocessing/graph_schema.py` |
| Active checkpoint path, current run state | `MEMORY.md` → Key Paths and Current State |
| Open bugs, known failure modes | `project_run8_audit_findings.md` |
| Training log field names | `ml/src/training/training_logger.py` |
| Training log data | `ml/logs/<run_name>/` (JSONL) |
| Run analysis, metric results | `docs/training/<run_name>-analysis.md` |
| Architecture decisions | `docs/ml/adr/INDEX.md` |
| Data module v2 config and gates | `docs/proposal/Data_Module_Proposals/README.md` |
| Interpretability experiment status | `docs/interpretability/EXPERIMENT_INDEX.md` |
| Threshold load path | `ml/src/inference/predictor.py` |
| Warmup and loss instantiation | `ml/src/training/trainer.py` |
| Eye-to-output relationship | `ml/src/models/sentinel_model.py` |

If the information you need is not in this table, find its canonical source file
before proceeding — do not assume or reconstruct from context.

---

## Rule 2 — Validate Your Validation

Every validation procedure must verify that it was performed correctly, not just
that it ran. There are three required layers.

### Layer 1 — Gate Assertions (write results, never just print)

Every step that produces a boolean result or a number must write its result
to a named file before the procedure continues. Format:

```
<check_name>: PASS | FAIL | UNVERIFIED — <one-line description>
Source: <file or tool that produced this result>
Date: <ISO date>
```

If a result cannot be confirmed from the tool output, mark it `UNVERIFIED`.
`UNVERIFIED` is not a pass. Document what prevented confirmation and stop
until it is resolved or explicitly accepted as a known gap.

### Layer 2 — Cross-Check (two independent sources)

For any metric or count that will be used to draw a conclusion (F1, threshold,
contamination count, label class counts, VRAM headroom), confirm it from two
independent sources where they exist. Sources might be: MLflow log vs. JSONL
epoch file, split index count vs. label file count, schema file vs. model config.

If the two sources disagree, that disagreement is itself a finding.
Stop, document it, and do not proceed until the discrepancy is resolved.

### Layer 3 — Completion Attestation (write before session ends)

At the end of any multi-step procedure, produce a structured attestation
and append it to the relevant doc (audit file, run analysis, or MEMORY.md).
Do not consider a procedure finished until the attestation is written.

```
## Procedure Attestation — <procedure name> — <ISO date>
Steps completed:   [list each step with PASS / FAIL / UNVERIFIED]
Steps skipped:     [list any skipped steps and explicit reason]
Unverified items:  [list anything marked UNVERIFIED]
New findings:      [link to doc entry, or "none"]
Written to:        [path of this attestation file]
```

A skipped step with no documented reason is treated as a gap.

---

## Rule 3 — No Floating Findings

Any finding, decision, anomaly, or open question discovered during a procedure
must be written to a named persistent file immediately — not at the end of the
session, not kept only in the conversation.

- New bug or failure mode → append to current audit doc (path in `MEMORY.md`)
- Architecture or data decision → create or append to relevant ADR
- Open question with no immediate answer → append to `## Open Questions` in the relevant doc
- Plan or intent agreed in conversation → write to the relevant spec or doc now

A finding that exists only in the conversation is considered lost.

---

## Rule 4 — Procedures Are Not Knowledge

Spec files contain *how to check* — not *what the answer is*.

- Never update a spec file to record a metric, value, or result
- Never add hardcoded run names, version numbers, or checkpoint paths to a spec file
- If you find yourself adding a specific value to a spec file, stop —
  that value belongs in `MEMORY.md`, a run analysis doc, or a config file
- Spec files must remain valid for future runs, future schema versions,
  and future checkpoints without modification

---

## Applying These Rules

Before starting any procedure from any spec file in this folder:

1. Confirm you have read `00_rules.md` in the current session
2. Read the relevant source files listed in Rule 1 for this task
3. Begin the procedure
4. Apply Layer 1 gate assertions at each verification step
5. Apply Layer 2 cross-checks on any metric that drives a decision
6. Write the Layer 3 attestation before the session ends or the task closes
