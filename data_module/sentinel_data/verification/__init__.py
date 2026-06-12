"""Verification submodule — Stage 4 of the SENTINEL v2 data pipeline.

Implements the BCCC-failure catcher: the verification stage that would have
caught the 89% Reentrancy FP and 86.9% CallToUnknown FP in the BCCC dataset.
Components: semantic_checker, tool_validator, fp_estimator, class_auditor,
negative_checker, probe_dataset, report_generator, gate.
"""
