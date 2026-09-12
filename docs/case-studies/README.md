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
3. [Version the Representation Instead of Patching History](03_version_the_representation_instead_of_patching_history.md) — preserving v9 for reproducibility while moving corrected external-call semantics into a separately accepted V10 lineage.
4. [Fail Closed on Unexplained Structural Drift](04_fail_closed_on_unexplained_structural_drift.md) — keeping physical acceptance false until every observed V10 transition in the exact candidate was independently reconciled.
5. [Promote a Selector Without Rewriting the Accepted Lineage](05_promote_a_selector_without_rewriting_the_accepted_lineage.md) — separating selector-policy evidence, physical candidate acceptance, and eventual training authority.
6. [Tool Silence Is Not a Clean Result](06_tool_silence_is_not_a_clean_result.md) — representing unavailable/degraded execution separately from successful zero-finding evidence in the AGENTS pipeline.
7. [A Valid Proof Does Not Prove the Whole Audit](07_a_valid_proof_does_not_prove_the_whole_audit.md) — separating proxy-proof integrity, V3 context/provenance attestation, registry persistence, and external transaction authority.

## Curation boundary

Seven cases are the intended P6 core package. Together they cover the strongest distinct evidence/engineering themes without turning this directory into a chronological rewrite of the repository:

- supervision truth;
- leakage/identity authority;
- representation versioning;
- fail-closed transition evidence;
- bounded selector promotion;
- degraded tool/evidence semantics;
- cryptographic-proof versus provenance/authority boundaries.

Add another case only when it introduces a materially different engineering decision that is useful to an external reviewer. Variants of the same R4 episode should stay in the primary evidence trail rather than becoming portfolio duplication.
