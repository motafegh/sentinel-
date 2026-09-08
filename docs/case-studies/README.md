# SENTINEL engineering case studies

These case studies curate a small set of engineering decisions that are useful to an external technical reviewer. They are not a replacement for source code, tests, the canonical handbook, or the R4 evidence/ADR chain.

Their purpose is to answer a practical question: **when SENTINEL encountered an uncomfortable engineering result, what evidence was available, what shortcut was rejected, what decision was made, and what limitation remained?**

## Reading and authority

Use this authority order for any disputed current claim:

1. executable source/config/tests;
2. committed machine-readable R4 policy/evidence/manifests;
3. canonical handbook;
4. accepted ADRs/decision records;
5. these case studies and other supplementary/historical material.

A case study therefore explains an accepted decision; it does not grant new training, model-quality, production, signing/broadcast, or ZK authority.

## Case-study format

Each case follows the same compact structure:

1. **Decision in one sentence**
2. **Problem**
3. **Evidence**
4. **Shortcut rejected**
5. **Decision**
6. **Implementation and validation**
7. **Result**
8. **Remaining limitation**
9. **Evidence trail**

This structure is intentionally evidence-first. A technically attractive outcome is not presented as established unless the repository's accepted evidence supports it.

## Published cases

1. [Unknown Is Not Negative](01_unknown_is_not_negative.md) — repairing supervision semantics before repaired retraining, rather than converting unsupported historical zeros into negative truth.
2. [Leakage Grouping Was an Evidence Problem](02_leakage_grouping_was_an_evidence_problem.md) — replacing convenient relationship-based grouping authority with versioned evidence-backed identity rules before relying on ML claims.

## Planned curation

Later P6 work may add only the strongest distinct decisions, such as V10 representation remediation, deterministic structural-drift reconciliation, guarded-selector promotion discipline, explicit refusal to manufacture unsupported evaluation metrics, AGENTS tool-execution semantics, and the ZK-proof versus V3-provenance boundary.

The goal is a small reviewable evidence package, not a chronological rewrite of the entire project history.
