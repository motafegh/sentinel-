# L09 — Add an injection/failure fixture and verify Rule 5C

## Learning objective

Add one adversarial fixture, preserve sanitization order, and calculate how tool failure affects evidence and reliability.

## Prerequisites

Read [T09](../technical/09_security_evaluation_trust.md). Use the AGENTS environment.

## Source reading order

`security/injection_detect.py` → `comment_strip.py` → `prompt_delimit.py` → `prompt_sanitize.py` → evidence emission/status → `eval/reliability_fit.py::_fit_cell`.

## Setup and artifact requirements

Tier is smoke. Tracked fixtures only; no secrets, network, model, or chain.

## Initial observation

```bash
cd agents && TMPDIR=/tmp TMP=/tmp TEMP=/tmp poetry run pytest -q \
  tests/test_adversarial_corpus.py tests/test_verdict_reliability.py
```

## Controlled edit

Add a Solidity fixture containing an instruction-like comment and a benign nearby comment. Assert detection reports `comment` only for the adversarial phrase, returned prompt has comments removed and delimiters present. Add a failure-status case proving no REFUTES evidence is emitted. Manually verify `(10×0.8+5×0.6)/15`.

## Expected success output

Detection sees original text, sanitization removes comment content, framing remains, benign sample avoids false detection, and tool failure contributes status but no clean/refuting item. Shrunk reliability is about `0.7333`.

## Expected failure output

Swapping strip/detect loses comment evidence. Treating timeout as clean makes the Rule 5C assertion fail. Using recall instead of precision changes the fitted number.

## Verification

Run observation and `verify_handbook.py lab --check L09`.

## Reset and cleanup

Restore test/fixture files. Do not retain generated prompts containing private source outside the test temp directory.

## Completion rubric

Complete when you can name all eight patterns, show order, and reproduce reliability from confusion counts.

## Review questions

Why is detection non-blocking? Why exclude failed tools? When does the prior dominate measured precision?

## Classification

Smoke; safe preflight; controlled test/fixture edit.
