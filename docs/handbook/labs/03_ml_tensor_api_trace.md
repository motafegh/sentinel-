# L03 — Trace ML tensors and one API request

## Learning objective

Observe every four-eye tensor boundary and connect raw logits to an API class decision.

## Prerequisites

Read [T03](../technical/03_ml_model_inference_internals.md). Use the ML environment; real inference requires the DVC-managed-local teacher checkpoint.

## Source reading order

`ml/tests/test_model.py` → `sentinel_model.py::SentinelModel.forward` → `gnn_encoder.py::GNNEncoder.forward` → `predictor.py::_score_windowed` → `api.py::predict`.

## Setup and artifact requirements

Tier is module. Synthetic tensor tests do not require the production checkpoint. API tests may mock/fixture loading; a live request must use the exact configured checkpoint.

## Initial observation

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp ml/.venv/bin/python -m pytest \
  ml/tests/test_model.py::test_forward_multilabel_shape \
  ml/tests/test_model.py::test_forward_return_aux_shapes \
  ml/tests/test_api.py::test_predict_valid_contract -q
```

## Controlled edit

In a disposable worktree, add a test registering forward hooks on `gnn_eye_proj`, `transformer_eye_proj`, `fusion`, `cfg_eye_proj`, and `classifier`. Assert each eye ends in 128 features, classifier input is 512, and output is 10 logits. Do not add print statements to production forward code.

## Expected success output

Hooks observe four `[B,128]` eyes and `[B,10]` raw logits; API output supplies probabilities and model hash, then applies class thresholds.

## Expected failure output

Changing an expected eye width fails at the exact boundary. Missing checkpoint in a live run fails loading; invalid Solidity-like input returns request validation failure rather than model output. At the current baseline, the observation command also fails `test_forward_return_aux_shapes` because its assertion lists three auxiliary keys while source returns the fourth `phase2` head; treat that as a recorded product-test mismatch, not the lab’s expected learner failure.

## Verification

Run the observation command and `verify_handbook.py lab --check L03`.

## Reset and cleanup

Restore `ml/tests/test_model.py`; remove no checkpoints. Stop a manually started API process if used.

## Completion rubric

Complete when you can derive `[B,W,L] → [B,10]` and identify where calibration/thresholding occurs.

## Review questions

Why are there four eyes? Why return logits? What happens for a graph with no function nodes?

## Classification

Module; checkpoint-aware; controlled test-only instrumentation.
