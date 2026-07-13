# 13 — Evaluation and release evidence

**Read this when:** you need to judge DATA quality, ML quality, AGENTS behavior, reliability weights, or release readiness.

**Skip this if:** you are only browsing architecture; do not skip it before promotion.

**Estimated reading time:** 15 minutes.

## 30-second summary

Evaluation has three layers. DATA proves provenance, label quality, leakage control, and representation integrity. ML measures ranking, thresholded decisions, calibration, OOD behavior, and interpretability. AGENTS measures whole-pipeline verdicts, paths, failures, evidence coverage, Fβ with β=2, nine behavioral gates, and per-tool reliability. A test pass means implementation consistency; release evidence means measured usefulness and safety.

## Just-enough mental model

```text
DATA gates: is the evidence set trustworthy?
      ↓
ML eval: does the model rank/calibrate/decide well?
      ↓
AGENTS eval: does the composed system route, fail, and report correctly?
      ↓
release record: artifact + commit + environment + metrics + gates + limitations
```

Never compare metrics unless corpus, split, class order, thresholds, tool availability, and deterministic/LLM mode are compatible.

## Actual runtime/source walkthrough

### DATA

DATA verification, split leakage, contamination benchmarks, feature distributions, co-occurrence, drift, overlap, and catalog diffs establish whether an export is safe to use. The relevant code is under [`verification`](../../data_module/sentinel_data/verification), [`splitting`](../../data_module/sentinel_data/splitting), [`analysis`](../../data_module/sentinel_data/analysis), and tracked [`benchmarks`](../../data_module/benchmarks).

### ML

Training validation records per-class and aggregate precision/recall/F1/AUC-PR plus calibration. Threshold tuning is per class. OOD suites, behavioral probes, cross-tool comparisons, reproducibility checks, and interpretability experiments provide complementary evidence. No single macro score replaces them.

### AGENTS

[`pipeline_metrics.py`](../../agents/src/eval/pipeline_metrics.py) computes per-class/macro metrics including Fβ, configured with β=2 to emphasize recall. [`gates.py`](../../agents/src/eval/gates.py) defines nine release assertions:

1. no consensus-flagged class silently becomes SAFE;
2. debate timeout emits INCONCLUSIVE in LLM mode;
3. no consensus vote disappears from final verdicts;
4. confidence 1.0 is not downgraded to SAFE;
5. vulnerable overall label is not paired with SAFE overall verdict;
6. zero false-positive verdicts on the safe subset;
7. the long-contract regression is detected;
8. eye predictions are present;
9. macro-F1 does not drop against the selected baseline.

[`run_benchmark.py`](../../agents/src/eval/run_benchmark.py) writes rows, metrics, gates, and reports. A run counts as passed only when its applicable gates pass; N/A conditions must remain visible.

### Reliability matrix and fitting

[`reliability_matrix.py`](../../agents/src/eval/reliability_matrix.py) builds per-tool/per-class TP/FP/FN/TN while excluding cases where `tool_status.ran` is not true. [`reliability_fit.py`](../../agents/src/eval/reliability_fit.py) computes measured precision `tp/(tp+fp)` and shrinks it toward the configured L1 prior:

```text
fitted = (n × measured_precision + alpha × prior) / (n + alpha)
n = tp + fp + fn + tn; alpha = 5
```

Zero-sample cells retain the prior. A fitted/prior delta of at least 0.05 requires a recorded justification. This is not the beta-posterior formula described in the superseded D1 plan.

## Interfaces, data shapes, and configuration

A release evidence bundle should name:

- commit, date, environment, deterministic/LLM/tool modes;
- DATA version, split, artifact hash, schema, contamination/leakage evidence;
- teacher checkpoint/hash, thresholds, calibration, proxy/circuit versions;
- per-class confusion and ranking/calibration metrics;
- Fβ β value, nine gate results, tool-status coverage, path/latency distributions;
- reliability matrix input and fitted config/justifications;
- known failures, skipped checks, local-only prerequisites, and comparison baseline.

The active decision policy is in [`verdicts_default.yaml`](../../agents/configs/verdicts_default.yaml); fitted reliability is in [`reliability_v3.yaml`](../../agents/configs/reliability_v3.yaml).

## Failure modes and current limitations

- Treating skipped/unavailable tools as negatives biases reliability and system metrics.
- Fβ emphasizes recall but does not constrain false positives without separate gates.
- Small benchmark corpora create high uncertainty; shrinkage reduces but does not remove it.
- LLM-on and deterministic/no-LLM runs are not directly interchangeable.
- Thresholds optimized on evaluation data invalidate that data as an unbiased final test.
- Product test counts are volatile and belong only in [current status](16_current_status.md).

## Common change recipe

For any policy or model change:

1. Freeze baseline, corpus, modes, and artifact identities.
2. State the expected metric/gate effect before implementation.
3. Run DATA validation if inputs changed, ML evaluation if model/thresholds changed, and AGENTS benchmark if system policy changed.
4. Compare per class and failure coverage, not only macro metrics.
5. Refit reliability only from valid `ran=true` evidence.
6. Record regressions and justifications; do not weaken a gate to obtain green output.

## Verification commands

```bash
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
cd agents
poetry run python -m src.eval.run_benchmark --help
poetry run python -m src.eval.reliability_fit --help
poetry run pytest -q -k 'eval or gate or reliability'
cd ..
python3 docs/handbook/tools/verify_handbook.py static
```

Full benchmark runs with real tools/LLMs are live checks and must preserve their raw evidence.

## Optional deep references

- [`docs/learning/08_evaluation_framework.md`](../learning/08_evaluation_framework.md)
- [`docs/learning/10_decision_numbers.md`](../learning/10_decision_numbers.md)
- [`agents/src/eval/README.md`](../../agents/src/eval/README.md)
- [ML training and quality](06_ml_training_quality.md)

## Technical mastery layer

### Prerequisite knowledge

Know confusion matrices, macro/micro metrics, Fβ, calibration, regression baselines, and statistical uncertainty.

### Source map and reading order

Read DATA verification/benchmarks, ML testing specs/promotion evidence, AGENTS `pipeline_metrics.py`, nine gates in `gates.py`, reliability matrix, and `reliability_fit.py::fit`. See [T09](technical/09_security_evaluation_trust.md).

### Execution trace and worked example

Labeled outcomes become per-class confusion cells. Measured reliability is precision `tp/(tp+fp)` and shrinks toward L1 prior with alpha 5. For `n=10`, measured `0.8`, prior `0.6`, fitted value is about `0.7333`. Gate failures remain release evidence even if an average metric improves.

### Implementation practice

[L04](labs/04_training_calibration_promotion.md) practices Fβ/threshold evidence; [L09](labs/09_injection_rule5c_reliability.md) practices reliability and failure exclusion. Freeze corpus identity before comparison.

### Review and ownership check

Can you reproduce every reported metric from counts and explain all nine gates without referring to volatile suite counts?
