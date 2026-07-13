# T09 — Security, trust, reliability, and evaluation internals

## Learning outcome

You can explain the exact sanitization order and pattern names, enforce Rule 5C under failures, compute reliability shrinkage, and read release gates without overstating ZK or provenance guarantees.

## Prerequisites

Read [Security and trust](../12_security_and_trust.md) and [Evaluation](../13_evaluation.md). Know confusion matrices, precision/recall/Fβ, prompt injection, and trust boundaries.

## Source map and reading order

1. `agents/src/security/injection_detect.py::detect_injections`.
2. `comment_strip.py`, `prompt_delimit.py`, then `prompt_sanitize.py::sanitize_for_prompt`.
3. Orchestration failure/status behavior and Rule 5C tests.
4. `agents/src/eval/{pipeline_metrics,gates,reliability_matrix,reliability_fit}.py`.
5. DATA verification/benchmark gates and ML testing/promotion evidence.

## Entry point and complete call chain

Before untrusted Solidity enters an LLM prompt, detection runs on the original source, then comments are stripped, then source is framed with delimiters. Detection reports eight names: comment, string, role-swap, extraction, identifier, NatSpec, multi, import. It records signals; it does not by itself prove malicious intent. During audit, tool failures update status. Rule 5C prevents unavailable/error/timeout outcomes from becoming clean or refuting evidence. Evaluation converts labeled outcomes into metrics/gates, builds per-source/class confusion cells, then fits reliability used by fusion.

## Important symbols and configuration

- Sanitization sequence is detect(original) → strip comments → delimit.
- Routing isolation prevents raw contract text or LLM behavior from selecting graph branches.
- Reliability measured value is precision `tp/(tp+fp)`.
- With sample count `n`, prior `p`, and `alpha=5`, fitted reliability is `(n×measured + alpha×p)/(n+alpha)`; zero samples use the prior verbatim.
- AGENTS evaluation has nine gates; module pages do not duplicate volatile run counts.

## Annotated source excerpt

Source: `agents/src/eval/reliability_fit.py::_fit_cell`

```python
denom = tp + fp
measured = (tp / denom) if denom > 0 else 0.0
if n == 0:
    fitted = prior
else:
    fitted = (n * measured + alpha * prior) / (n + alpha)
```

The prior contributes five pseudo-observations at the default alpha; this is not a recall formula and not a direct average of TP/TN accuracy.

## Worked example

A source/class cell has `tp=8`, `fp=2`, so measured precision is `0.8`; `n=10`, configured L1 prior `0.6`, alpha `5`. Fitted reliability is `(10×0.8 + 5×0.6)/15 = 11/15 ≈ 0.7333`. If the tool timed out on another contract, that outcome is excluded from clean/refute evidence under Rule 5C; it should be represented in tool status and evaluation availability.

## Success trace

All injection patterns are detected on untouched input; prompt receives stripped/framed source; routing remains source/LLM isolated; failures are excluded from negative evidence; confusion cells reconcile; fitted values retain prior/provenance metadata; all required release gates produce explicit pass/fail evidence.

## Failure trace

Stripping before detection erases comment/NatSpec evidence. Treating no output as “clean” biases safety and reliability. Fitting against unlabeled or leaked evaluation data invalidates weights. A valid proxy proof cannot repair unsigned provenance or prove teacher execution. Secret-shaped environment values must never enter reports/docs.

## Design reasoning and rejected alternatives

Layered prompt defense preserves detection telemetry while reducing instructions visible to the LLM. Deterministic routing limits prompt-influenced control flow. Shrinkage prevents tiny samples from producing extreme reliabilities. Gates encode non-negotiable behaviors that aggregate metrics can hide.

## Safe change walkthrough

To add an injection pattern, create a positive and benign-near-match fixture, implement detector, preserve original-source ordering, update pattern registry/docs, and run adversarial and routing tests. To refit reliability, freeze labeled corpus identity, rebuild matrix, inspect zero/low-sample cells and drop gates, then version fitted output; do not hand-edit production weights.

## Guided lab

Complete [L09 — injection, Rule 5C, and reliability](../labs/09_injection_rule5c_reliability.md).

## Tests and expected results

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_adversarial_corpus.py tests/test_routing_isolation.py \
  tests/test_verdict_reliability.py
```

Expected: named patterns, isolation, fallback, and reliability behavior pass. Release-gate failures remain visible; do not weaken a gate to produce green output.

## Review questions

Why detect before strip? How does Rule 5C affect a confusion matrix? Compute shrinkage for zero samples. Which security assertions remain off-chain?

## Ownership checklist

- I can name all eight patterns and execution order.
- I never turn absence/failure into safety evidence.
- I can reproduce a fitted weight from counts and prior.
- I separate cryptographic, operational, and operator-asserted trust.
